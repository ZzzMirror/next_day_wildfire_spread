# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Earth Engine helper functions.

Details on the Earth Engine Data Catalog can be found here:
https://developers.google.com/earth-engine/datasets

In order to use this library you need to authenticate and initialize the
Earth Engine library.
"""

from typing import List, Text, Tuple

import ee
import os
import tensorflow as tf

from data_export import ee_utils


def _get_all_feature_bands():
  """Returns list of all bands corresponding to features."""
  return (ee_utils.DATA_BANDS[ee_utils.DataType.ELEVATION_SRTM] +
          ['population'] +
          ee_utils.DATA_BANDS[ee_utils.DataType.DROUGHT_GRIDMET] +
          ee_utils.DATA_BANDS[ee_utils.DataType.VEGETATION_VIIRS] +
          ee_utils.DATA_BANDS[ee_utils.DataType.WEATHER_GRIDMET] +
          ['PrevFireMask'])


def _get_all_response_bands():
  """Returns list of all bands corresponding to labels."""
  return ee_utils.DATA_BANDS[ee_utils.DataType.FIRE_MODIS]


def _add_index(i, bands):
  """Appends the index number `i` at the end of each element of `bands`."""
  return [f'{band}_{i}' for band in bands]


def _get_all_image_collections():
  """Gets all the image collections and corresponding time sampling."""
  image_collections = {
      'drought':
          ee_utils.get_image_collection(ee_utils.DataType.DROUGHT_GRIDMET),
      'vegetation':
          ee_utils.get_image_collection(ee_utils.DataType.VEGETATION_VIIRS),
      'weather':
          ee_utils.get_image_collection(ee_utils.DataType.WEATHER_GRIDMET),
      'fire':
          ee_utils.get_image_collection(ee_utils.DataType.FIRE_MODIS),
  }
  time_sampling = {
      'drought':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.DROUGHT_GRIDMET],
      'vegetation':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.VEGETATION_VIIRS],
      'weather':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.WEATHER_GRIDMET],
      'fire':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.FIRE_MODIS],
  }
  return image_collections, time_sampling


def _verify_feature_collection(
    feature_collection
):
  """Verifies the feature collection is valid.

  If the feature collection is invalid, resets the feature collection.

  Args:
    feature_collection: An EE feature collection.

  Returns:
    `(feature_collection, size)` a tuple of the verified feature collection and
    its size.
  """
  try:
    size = int(feature_collection.size().getInfo())
  except ee.EEException:
    # Reset the feature collection
    feature_collection = ee.FeatureCollection([])
    size = 0
  return feature_collection, size


def _get_time_slices(
    window_start,
    window,
    projection,  # Defer calling until called by test code
    resampling_scale,
    lag = 0,
):
  """Extracts the time slice features.

  Args:
    window_start: Start of the time window over which to extract data.
    window: Length of the window (in days).
    projection: projection to reproject all data into.
    resampling_scale: length scale to resample data to.
    lag: Number of days before the fire to extract the features.

  Returns:
    A list of the extracted EE images.
  """
  image_collections, time_sampling = _get_all_image_collections()
  window_end = window_start.advance(window, 'day')
  drought = image_collections['drought'].filterDate(
      window_start.advance(-lag - time_sampling['drought'], 'day'),
      window_start.advance(
          -lag, 'day')).median().reproject(projection).resample('bicubic')
  vegetation = image_collections['vegetation'].filterDate(
      window_start.advance(-lag - time_sampling['vegetation'], 'day'),
      window_start.advance(
          -lag, 'day')).median().reproject(projection).resample('bicubic')
  weather = image_collections['weather'].filterDate(
      window_start.advance(-lag - time_sampling['weather'], 'day'),
      window_start.advance(-lag, 'day')).median().reproject(
          projection.atScale(resampling_scale)).resample('bicubic')
  prev_fire = image_collections['fire'].filterDate(
      window_start.advance(-lag - time_sampling['fire'], 'day'),
      window_start.advance(-lag, 'day')).map(
          ee_utils.remove_mask).max().rename('PrevFireMask')
  fire = image_collections['fire'].filterDate(window_start, window_end).map(
      ee_utils.remove_mask).max()
  detection = fire.clamp(6, 7).subtract(6).rename('detection')
  return [drought, vegetation, weather, prev_fire, fire, detection]


def _export_dataset(
    bucket,
    folder,
    prefix,
    start_date,
    start_days,
    geometry,
    kernel_size,
    sampling_scale,
    num_samples_per_file,
):
  """Exports the dataset TFRecord files for wildfire risk assessment.

  Args:
    bucket: Google Cloud bucket
    folder: Folder to which to export the TFRecords.
    prefix: Export file name prefix.
    start_date: Start date for the EE data to export.
    start_days: Start day of each time chunk to export.
    geometry: EE geometry from which to export the data.
    kernel_size: Size of the exported tiles (square).
    sampling_scale: Resolution at which to export the data (in meters).
    num_samples_per_file: Approximate number of samples to save per TFRecord
      file.
  """

  def _verify_and_export_feature_collection(
      num_samples_per_export,
      feature_collection,
      file_count,
      features,
  ):
    """Wraps the verification and export of the feature collection.

    Verifies the size of the feature collection and triggers the export when
    it is larger than `num_samples_per_export`. Resets the feature collection
    and increments the file count at each export.

    Args:
      num_samples_per_export: Approximate number of samples per export.
      feature_collection: The EE feature collection to export.
      file_count: The TFRecord file count for naming the files.
      features: Names of the features to export.

    Returns:
      `(feature_collection, file_count)` tuple of the current feature collection
        and file count.
    """
    feature_collection, size_count = _verify_feature_collection(
        feature_collection)
    if size_count > num_samples_per_export:
      ee_utils.export_feature_collection(
          feature_collection,
          description=prefix + '_{:03d}'.format(file_count),
          bucket=bucket,
          folder=folder,
          bands=features,
      )
      file_count += 1
      feature_collection = ee.FeatureCollection([])
    return feature_collection, file_count

  elevation = ee_utils.get_image(ee_utils.DataType.ELEVATION_SRTM)
  end_date = start_date.advance(max(start_days), 'days')
  population = ee_utils.get_image_collection(ee_utils.DataType.POPULATION)
  # Could also move to using the most recent population data for a given sample,
  # which requires more EE logic.
  population = population.filterDate(start_date,
                                     end_date).median().rename('population')
  projection = ee_utils.get_image_collection(ee_utils.DataType.WEATHER_GRIDMET)
  projection = projection.first().select(
      ee_utils.DATA_BANDS[ee_utils.DataType.WEATHER_GRIDMET][0]).projection()
  resampling_scale = (
      ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_GRIDMET])

  all_days = []
  for day in start_days:
    for i in range(7):
      all_days.append(day + i)

  window = 1
  sampling_limit_per_call = 60
  features = _get_all_feature_bands() + _get_all_response_bands()

  file_count = 0
  feature_collection = ee.FeatureCollection([])
  for start_day in all_days:
    window_start = start_date.advance(start_day, 'days')
    time_slices = _get_time_slices(window_start, window, projection,
                                   resampling_scale)
    image_list = [elevation, population] + time_slices[:-1]
    detection = time_slices[-1]
    arrays = ee_utils.convert_features_to_arrays(image_list, kernel_size)
    to_sample = detection.addBands(arrays)

    fire_count = ee_utils.get_detection_count(
        detection,
        geometry=geometry,
        sampling_scale=10 * sampling_scale,
    )
    if fire_count > 0:
      samples = ee_utils.extract_samples(
          to_sample,
          detection_count=fire_count,
          geometry=geometry,
          sampling_ratio=0,  # Only extracting examples with fire.
          sampling_limit_per_call=sampling_limit_per_call,
          resolution=sampling_scale,
      )
      feature_collection = feature_collection.merge(samples)

      feature_collection, file_count = _verify_and_export_feature_collection(
          num_samples_per_file, feature_collection, file_count, features)
  # Export the remaining feature collection
  _verify_and_export_feature_collection(0, feature_collection, file_count,
                                        features)


def export_ml_datasets(
    bucket,
    folder,
    start_date,
    end_date,
    prefix = '',
    kernel_size = 128,
    sampling_scale = 1000,
    eval_split_ratio = 0.125,
    num_samples_per_file = 1000,
):
  """Exports the ML dataset TFRecord files for wildfire risk assessment.

  Export is to Google Cloud Storage.

  Args:
    bucket: Google Cloud bucket
    folder: Folder to which to export the TFRecords.
    start_date: Start date for the EE data to export.
    end_date: End date for the EE data to export.
    prefix: File name prefix to use.
    kernel_size: Size of the exported tiles (square).
    sampling_scale: Resolution at which to export the data (in meters).
    eval_split_ratio: Split ratio for the divide between training and evaluation
      datasets.
    num_samples_per_file: Approximate number of samples to save per TFRecord
      file.
  """

  split_days = ee_utils.split_days_into_train_eval_test(
      start_date, end_date, split_ratio=eval_split_ratio, window_length_days=8)

  for mode in ['train', 'eval', 'test']:
    sub_prefix = f'{mode}_{prefix}'
    _export_dataset(
        bucket=bucket,
        folder=folder,
        prefix=sub_prefix,
        start_date=start_date,
        start_days=split_days[mode],
        geometry=ee.Geometry.Rectangle(ee_utils.COORDINATES['US']),
        kernel_size=kernel_size,
        sampling_scale=sampling_scale,
        num_samples_per_file=num_samples_per_file)


def export_ml_datasets_local(
    output_folder,
    start_date,
    end_date,
    prefix='',
    kernel_size=128,
    sampling_scale=1000,
    eval_split_ratio=0.125,
    num_samples_per_file=1000,
):
    """Exports the ML dataset TFRecord files for wildfire risk assessment to a local folder.

    Args:
        output_folder: Local folder path to which to export the TFRecords.
        start_date: Start date for the EE data to export.
        end_date: End date for the EE data to export.
        prefix: File name prefix for output files.
        kernel_size: Size of the exported tiles (square).
        sampling_scale: Resolution at which to export the data (in meters).
        eval_split_ratio: Split ratio for the divide between training and evaluation datasets.
        num_samples_per_file: Approximate number of samples to save per TFRecord file.
    """
    # 将日期范围划分为训练、评估和测试集
    split_days = ee_utils.split_days_into_train_eval_test(
        start_date, end_date, split_ratio=eval_split_ratio, window_length_days=8)

    for mode in ['train', 'eval', 'test']:
        sub_prefix = f'{mode}_{prefix}'
        _export_dataset_local(
            output_folder=output_folder,
            prefix=sub_prefix,
            start_date=start_date,
            start_days=split_days[mode],
            geometry=ee.Geometry.Rectangle(ee_utils.COORDINATES['US']),
            kernel_size=kernel_size,
            sampling_scale=sampling_scale,
            num_samples_per_file=num_samples_per_file)

# 定义数据提取函数
def extract_data(window_start):
    # 根据起始日期提取相应的图像数据
    image = ee.ImageCollection("MODIS/006/MOD13Q1").filterDate(window_start, window_start.advance(1, 'days')).first()
    # 进行数据预处理，例如裁剪、缩放等操作
    # ...

    return image


def _export_dataset_local(
    output_folder,
    prefix,
    start_date,
    start_days,
    geometry,
    kernel_size,
    sampling_scale,
    num_samples_per_file,
):
    """Exports the dataset TFRecord files for wildfire risk assessment to a local folder.

    Args:
        output_folder: Local folder path to which to export the TFRecords.
        prefix: Export file name prefix.
        start_date: Start date for the EE data to export.
        start_days: Start day of each time chunk to export.
        geometry: EE geometry from which to export the data.
        kernel_size: Size of the exported tiles (square).
        sampling_scale: Resolution at which to export the data (in meters).
        num_samples_per_file: Approximate number of samples to save per TFRecord file.
    """



    # 获取需要的图像数据，例如海拔、人口密度等
    elevation = ee_utils.get_image(ee_utils.DataType.ELEVATION_SRTM)
    vegetation = ee_utils.get_image(ee_utils.DataType.VEGETATION_VIIRS)
    drought = ee_utils.get_image(ee_utils.DataType.DROUGHT_GRIDMET)
    weather = ee_utils.get_image(ee_utils.DataType.WEATHER_ERA5)
    fire = ee_utils.get_image(ee_utils.DataType.FIRE_MODIS)
    population = ee_utils.get_image(ee_utils.DataType.POPULATION)

    # 定义裁剪图像函数
    def clip_image(image, region):
        return image.clip(region)

    # 定义缩放图像函数
    def rescale_image(image, scale):
        return image.reproject(crs='EPSG:4326', scale=scale)

    # 定义数据提取函数
    def extract_data(window_start):
        # 根据起始日期提取相应的图像数据
        image = ee.ImageCollection("MODIS/006/MOD13Q1").filterDate(window_start,
                                                                   window_start.advance(1, 'days')).first()

        # 定义感兴趣区域的范围（这里以美国为例）
        region = ee.Geometry.Rectangle([-125, 25, -65, 50])

        # 对图像数据进行裁剪和缩放操作
        clipped_image = clip_image(image, region)
        scaled_image = rescale_image(clipped_image, 1000)  # 将分辨率缩放到 1000 米

        return scaled_image

    # 循环遍历每个开始日期，并将数据导出到 TFRecord 文件

    for start_day in start_days:
        # 循环遍历每个开始日期，并将数据导出到本地文件
        window_start = start_date.advance(start_day, 'days')
        data = extract_data(window_start)

        # 将数据保存为 TFRecord 格式的本地文件
        file_path = os.path.join(output_folder, f'{prefix}_{start_day}.tfrecord')

        #print('导出的tfr文件路径：'+file_path)
        write_data_to_tfrecord(file_path, data)  # 将数据写入 TFRecord 文件

def write_data_to_tfrecord(file_path, data):
    """将数据写入 TFRecord 文件。

    Args:
        file_path: TFRecord 文件的路径。
        data: 要写入的数据,data是image。
    """

    with tf.io.TFRecordWriter(file_path) as writer:
        # 提取图像数据
        # 这里假设 data 是一个包含图像数据的对象
        image_data = data.serialize()

        # 将图像数据写入 TFRecord 文件
        writer.write(image_data)
    '''
    with tf.io.TFRecordWriter(file_path) as writer:
        # 将数据序列化并写入 TFRecord 文件
        serialized_data = serialize_data(data)
        #print('序列化数据导出的tfr文件路径：'+file_path)
        writer.write(serialized_data)
    '''

def serialize_data(data):
    """序列化数据。

    Args:
        data: 要序列化的数据。

    Returns:
        序列化后的数据。
    """
    # 这里假设 data 是一个字典，包含需要的数据，需要根据实际情况进行修改
    feature = {
        'elevation': _float_feature(data['elevation']),
        'vegetation': _int64_feature(data['vegetation']),
        'drought': _float_feature(data['drought']),
        'weather': _int64_feature(data['weather']),
        'fire': _float_feature(data['fire']),
        'population': _int64_feature(data['population']),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _int64_feature(value):
    """生成一个 int64 特征。

    Args:
        value: 要转换的值。

    Returns:
        int64 特征。
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """生成一个 float 特征。

    Args:
        value: 要转换的值。

    Returns:
        float 特征。
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
