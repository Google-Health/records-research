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
"""Functions to build a tf.contrib.learn.Experiment."""
import config_utils
import data_provider
import experiment_config_pb2
import models
import tensorflow as tf

from google.protobuf import text_format


def get_experiment_fn(experiment_config: experiment_config_pb2.ExperimentConfig,
                      warm_start_from=None,
                      **kwargs):
  """Creates a function which builds a tf.contrib.learn.Experiment object.

  Args:
    experiment_config: experiment configuration.
    warm_start_from: Optional string filepath to a checkpoint or SavedModel to
      warm-start from.
    **kwargs: Additional keyword arguments passed to experiment.

  Returns:
    A function of two arguments, `run_config` and `hparams`, which builds a
    tf.contrib.learn.Experiment object.
  """

  def experiment_fn(run_config, hparams: tf.HParams = None):
    """Create a tf.contrib.learn.Experiment object.

    Args:
      run_config: An instance of learn_runner.EstimatorConfig.
      hparams: hparams passed from FLAGs.hparams which will be used to override
        the values from experiment_config.

    Returns:
      A tf.contrib.learn.Experiment object.
    """
    config = experiment_config
    if hparams is not None:
      # Populate command-line hparams to config.
      tf.logging.info('Using command-line hyperparameters %s', hparams)
      config = config_utils.merge_from_hparams(
          experiment_config, hparams, delimiter='__')
    config.experiment_dir = run_config.model_dir
    tf.logging.info('experiment config: ' + text_format.MessageToString(config))
    tf.logging.info('run config:' +
                    text_format.MessageToString(run_config.tf_config))
    estimator = tf.estimator.Estimator(
        model_fn=models.build_model_ops,
        params=config.model,
        config=run_config,
        warm_start_from=warm_start_from)
    provider = data_provider.DataProvider.from_config(config)
    export_strategies = []
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=provider.get_input_fn(tf.estimator.ModeKeys.TRAIN),
        eval_input_fn=provider.get_input_fn(tf.estimator.ModeKeys.EVAL),
        export_strategies=export_strategies,
        **kwargs)

  return experiment_fn
