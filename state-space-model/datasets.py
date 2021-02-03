# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A slim-style dataset for clinical time series dataset."""

import dataset
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder


def strip_raw_feature(code):
  if code.endswith('_raw'):
    code = code[:-4]
  return code


class ClinicalSeriesDataset(dataset.Dataset):
  """Clinical time series dataset."""

  def __init__(self, data_sources, has_mask, labels, context_window_size,
               observation_codes, intervention_codes):
    """Creates a dataset for clinical time series dense feature lab data.

    Args:
      data_sources: A list of files/patterns for the slim Dataset.
      has_mask: Whether the dataset has obs_mask and true_length_hr feature.
      labels: A list of labels in string, corresponding to labels in
        ModelConfig.
      context_window_size: size of the context window, i.e, the length of the
        time series.
      observation_codes: A list of features corresponding to the observation
        time series data.
      intervention_codes: A list of features corresponding to the intervention
        time series data.

    Returns:
      A slim dataset with proper reader and decoders.
    """

    keys_to_features = {}
    items_to_handlers = {}

    keys_to_features['context_window_size'] = tf.FixedLenFeature(
        [], dtype=tf.int64, default_value=0)
    items_to_handlers[
        'context_window_size'] = tfexample_decoder.Tensor(
            'context_window_size', default_value=0)

    keys_to_features['context_window_start_time_sec'] = tf.FixedLenFeature(
        [], dtype=tf.int64, default_value=0)
    items_to_handlers[
        'context_window_start_time_sec'] = tfexample_decoder.Tensor(
            'context_window_start_time_sec', default_value=0)

    keys_to_features['trigger_time_sec'] = tf.FixedLenFeature(
        [], dtype=tf.int64, default_value=0)
    items_to_handlers['trigger_time_sec'] = tfexample_decoder.Tensor(
        'trigger_time_sec', default_value=0)

    if has_mask:
      keys_to_features['true_length_hr'] = tf.FixedLenFeature([],
                                                              dtype=tf.int64,
                                                              default_value=0)
      items_to_handlers['true_length_hr'] = tfexample_decoder.Tensor(
          'true_length_hr', default_value=0)

    tf.logging.info('Labels are:')
    for label in labels:
      tf.logging.info(label)
      keys_to_features[label[0]] = tf.FixedLenFeature([],
                                                      dtype=tf.int64,
                                                      default_value=-1)
      items_to_handlers[label[0]] = tfexample_decoder.Tensor(label[0])
      # This label is for a survival analysis event.
      if label[1]:
        tf.logging.info(label[0] + '.time_of_event')
        keys_to_features[label[0] + '.time_of_event'] = tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=0)
        items_to_handlers[label[0] +
                          '.time_of_event'] = tfexample_decoder.Tensor(
                              label[0] + '.time_of_event')

    tf.logging.info('Features are:')
    for observation in observation_codes:
      tf.logging.info(observation)
      keys_to_features[observation] = tf.FixedLenFeature(
          shape=[context_window_size], dtype=tf.float32)
      items_to_handlers[observation] = tfexample_decoder.Tensor(
          observation, default_value=-1)
      if has_mask:
        observation = strip_raw_feature(observation)
        keys_to_features[observation + '_mask'] = tf.FixedLenFeature(
            shape=[context_window_size], dtype=tf.float32)
        items_to_handlers[observation + '_mask'] = tfexample_decoder.Tensor(
            observation + '_mask', default_value=0)

    for intervention in intervention_codes:
      tf.logging.info(intervention)
      keys_to_features[intervention] = tf.FixedLenFeature(
          shape=[context_window_size], dtype=tf.float32)
      items_to_handlers[intervention] = tfexample_decoder.Tensor(
          intervention, default_value=-1)
      if has_mask:
        intervention = strip_raw_feature(intervention)

        keys_to_features[intervention + '_mask'] = tf.FixedLenFeature(
            shape=[context_window_size], dtype=tf.float32)
        items_to_handlers[intervention + '_mask'] = tfexample_decoder.Tensor(
            intervention + '_mask', default_value=0)

    decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                 items_to_handlers)

    super(ClinicalSeriesDataset, self).__init__(
        data_sources=data_sources,
        reader=tf.compat.v1.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})
