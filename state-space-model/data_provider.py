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

"""Data provider which prepares tensor inputs based on slim API."""

from typing import List, Optional, Tuple, Dict

import datasets
import experiment_config_pb2
from six import iteritems
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader


def _data_zip(train_data, eval_data, test_data):
  """Zip train, eval, test data."""
  return {
      tf.estimator.ModeKeys.TRAIN: train_data,
      tf.estimator.ModeKeys.EVAL: eval_data,
      tf.estimator.ModeKeys.INFER: test_data,
  }
# Scaling factor of queue size to batch size
QUEUE_SCALING_FACTOR = 10


class DataProvider(object):
  """Data provider for EMR observational data.

  Data provider provides a method 'get_input_fn', which constructs batched
  tensors as input to a tf.contrib.learn.Estimator.
  """

  def __init__(self,
               train_data: tf.contrib.slim.dataset.Dataset = None,
               eval_data: tf.contrib.slim.dataset.Dataset = None,
               test_data: tf.contrib.slim.dataset.Dataset = None,
               feature_keys: List[bytes] = None,
               label_keys: List[bytes] = None,
               batch_size: int = 32,
               queue_capacity: int = 256,
               queue_min: int = 128,
               num_readers: int = 1):
    """Construct a data provider.

    Args:
      train_data: Training data.
      eval_data: Evaluation data.
      test_data: Test data.
      feature_keys: List of tensorflow.Example feature keys.
      label_keys: List of tensorflow.Example feature keys which are labels for
        training.
      batch_size: Batch size.
      queue_capacity: Capacity of the queue.
      queue_min: The minimum number of records after dequeue.
      num_readers: The number of parallel readers.

    Raises:
      ValueError: If `feature_keys` is not supplied.
    """
    if feature_keys is None:
      raise ValueError('`feature_keys` must be supplied.')

    self._feature_keys = feature_keys
    self._label_keys = label_keys
    self._batch_size = batch_size
    self._queue_capacity = queue_capacity
    self._queue_min = queue_min
    self._num_readers = num_readers
    self._data = _data_zip(train_data, eval_data, test_data)

  @classmethod
  def from_config(cls, config: experiment_config_pb2.ExperimentConfig):
    """Constructs a data provider from an experiment config.

    Args:
      config: Experiment configuration.

    Returns:
      An instance of data provider.
    """
    labels = [
        (label.name, label.is_survival_event) for label in config.model.labels
    ]

    train_data = datasets.ClinicalSeriesDataset(
        data_sources=list(config.train_sources),
        has_mask=config.model.has_mask,
        labels=labels,
        context_window_size=config.model.context_window_size,
        observation_codes=config.model.observation_codes,
        intervention_codes=config.model.intervention_codes)

    eval_data = datasets.ClinicalSeriesDataset(
        data_sources=list(config.eval_sources),
        has_mask=config.model.has_mask,
        labels=labels,
        context_window_size=config.model.context_window_size,
        observation_codes=config.model.observation_codes,
        intervention_codes=config.model.intervention_codes)

    test_data = datasets.ClinicalSeriesDataset(
        data_sources=list(config.test_sources),
        has_mask=config.model.has_mask,
        labels=labels,
        context_window_size=config.model.context_window_size,
        observation_codes=config.model.observation_codes,
        intervention_codes=config.model.intervention_codes)

    feature_keys = list(config.model.observation_codes)
    feature_keys = feature_keys + list(config.model.intervention_codes)
    if config.model.has_mask:
      for code in feature_keys:
        code = datasets.strip_raw_feature(code)
        feature_keys = feature_keys + [code + '_mask']
      feature_keys = feature_keys + ['true_length_hr']

    feature_keys = feature_keys + ['context_window_size']
    feature_keys = feature_keys + ['context_window_start_time_sec']

    return cls(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=config.model.batch_size,
        feature_keys=feature_keys,
        label_keys=labels)

  def _get_tensor_and_example(
      self,
      mode: tf.estimator.ModeKeys,
      shuffle: bool = False,
      num_epochs: Optional[int] = None) -> Tuple[Dict[bytes, tf.Tensor], bytes]:
    """Read and decode the serialized tf.Example into tensors.

    Args:
      mode: One of tf.estimator.ModeKeys {TRAIN,EVAL,INFER}.
      shuffle: Whether to shuffle the input.
      num_epochs: Number of times a tf.Example will be visited in generating the
        input. If set to None, each Example will be cycled indefinitely.

    Returns:
      Tuple with:
      A dictionary that maps tensorflow.Example feature names to tensors.
      serialized_example: bytes, a serialized example.
    """
    dataset = self._data[mode]
    if mode == tf.estimator.ModeKeys.INFER:
      serialized_example = tf.placeholder(
          dtype=tf.string, shape=[], name='input_serialized_examples')
    else:
      _, serialized_example = parallel_reader.parallel_read(
          dataset.data_sources,
          reader_class=dataset.reader,
          num_epochs=num_epochs,
          num_readers=self._num_readers,
          shuffle=shuffle,
          capacity=self._queue_capacity,
          min_after_dequeue=self._queue_min)
    items = dataset.decoder.list_items()
    tensors = dataset.decoder.decode(serialized_example, items)

    return dict(zip(items, tensors)), serialized_example

  def get_input_fn(self,
                   mode: tf.estimator.ModeKeys,
                   shuffle: Optional[bool] = None,
                   num_epochs: Optional[int] = None):
    """Get an input function for the given mode.

    Args:
      mode: One of tf.estimator.ModeKeys {TRAIN,EVAL,INFER}.
      shuffle: Whether shuffle the input. If shuffle is None, only data under
        TRAIN mode is shuffled.
      num_epochs: Number of times a tensorflow.Example will be visited in
        generating the input. If set to None, each Example will be cycled
        indefinitely.

    Returns:
      A callable that returns a pair of dictionaries containing
      the batched tensors for 'features' and 'labels', respectively.
      labels tensors have shape [batch_size, 1]
      features tensors have shape [batch_size]

    Raises:
      ValueError: A 'mode' is supplied for which there is no data.
    """
    if not self._data[mode]:
      raise ValueError('No data provided for mode %s' % mode)

    def input_fn():
      """An input function under training/eval mode."""
      with tf.name_scope('ReadData'):
        tensor, _ = self._get_tensor_and_example(
            mode, shuffle=shuffle, num_epochs=num_epochs)

      with tf.name_scope('Batch'):
        batched = tf.train.batch(
            tensor,
            batch_size=self._batch_size,
            dynamic_pad=True,
            capacity=QUEUE_SCALING_FACTOR * self._batch_size)

      features = {key: batched[key] for key in self._feature_keys}
      labels = {}
      for label_tuple in self._label_keys:
        labels[label_tuple[0]] = tf.expand_dims(
            batched[label_tuple[0]], axis=-1)
        if label_tuple[1]:
          labels[label_tuple[0] + '.time_to_event'] = tf.expand_dims(
              batched[label_tuple[0] + '.time_of_event'] -
              batched['trigger_time_sec'],
              axis=-1)

      return features, labels

    def serving_input_fn():
      """Construct a ServingInputReceiver function under infer mode."""
      input_keys = tf.placeholder(
          dtype=tf.string, shape=[None], name='input_keys')

      with tf.name_scope('ReadInferenceData'):
        feature_dict, serialized_example = self._get_tensor_and_example(
            mode, shuffle=shuffle, num_epochs=num_epochs)

      # At serving time, the batch size will be 1. We need to reshape the
      # features to account for this.
      features = {}
      for key, tensor in iteritems(feature_dict):
        features[key] = tf.expand_dims(tensor, 0)
        tf.logging.info(key)
      tf.logging.info(features)

      inputs = {
          'input_keys': input_keys,
          'input_examples': serialized_example,
      }

      return tf.estimator.export.ServingInputReceiver(
          features=features, receiver_tensors=inputs)

    if shuffle is None:
      shuffle = (mode == tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.INFER:
      return serving_input_fn
    else:
      return input_fn
