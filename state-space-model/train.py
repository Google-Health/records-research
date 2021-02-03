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

"""Main executable for training and eval."""
import os
import re

from absl import flags
import config_utils
import experiment
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner

flags.DEFINE_string('hparams', '', 'hparams')

flags.DEFINE_string('experiment_config', None,
                    'pbtxt file containing ExperimentConfig.')

flags.DEFINE_string('train_path', None, 'path to train data')

flags.DEFINE_string('eval_path', None, 'path to eval data')
flags.DEFINE_string('warm_start_from', None,
                    'Optional string filepath to a checkpoint or SavedModel to '
                    'warm-start from.')

flags.DEFINE_integer(
    'num_train_steps', None,
    'The number of steps to run training for. None means continuous training.')

flags.DEFINE_integer('num_eval_steps', 200,
                     'The number of steps to run evaluation for.')

flags.DEFINE_integer('save_checkpoints_steps', 1000,
                     'The number of training steps to save a checkpoint for.')

flags.DEFINE_integer('keep_checkpoint_max', 50,
                     'The number of training steps to save a checkpoint for.')

flags.DEFINE_integer(
    'continuous_eval_throttle_secs', 20,
    'Number of seconds between evaluations during a continuous eval job.')

FLAGS = flags.FLAGS


def default_hparams():
  """Define default Hparams."""

  hparam_kwargs = {
      'model__learning_rate': 0.001,
      'model__da_state': 100,
      'model__da_tlen': 240,
      'model__da_sslot': 12,
      'model__da_unit': 32,
      'model__da_psi_mlpl': 1,
      'model__da_phi_mlpl': 3,
      'model__ds_state': 30,
      'model__ds_nrl': 1,
      'model__rnn_type': 'basic',
      'model__rnn_cell_type': 'lstm',
      'model__last_obs_len': 30,
      'model__sys_id_weight': 1.0,
  }
  return tf.HParams(**hparam_kwargs)


def get_best_model_dir(model_dir):
  """Get the best model dir from model_dir.

  Args:
    model_dir: Model dir.

  Returns:
    A string for best model dir.
  """
  if model_dir is None:
    return None
  paths = os.listdir(model_dir)

  versions = [int(p) for p in paths if re.fullmatch(r'\d+', p)]
  best_model_dir = '%s/%d' % (model_dir, max(versions))
  return best_model_dir


def main(_):
  # Parse hparams from FLAGs. Format example is provided below.
  # --hparams="model__optimizer__learning_rate=0.1,model__min_kernel_size=3"
  hparams = default_hparams().parse(FLAGS.hparams)
  experiment_config = config_utils.load_experiment_config(
      FLAGS.experiment_config)
  if FLAGS.train_path is not None:
    experiment_config.train_sources[0] = FLAGS.train_path
  if FLAGS.eval_path is not None:
    experiment_config.eval_sources[0] = FLAGS.eval_path

  best_model_dir = get_best_model_dir(FLAGS.warm_start_from)
  experiment_fn = experiment.get_experiment_fn(
      experiment_config,
      warm_start_from=best_model_dir,
      train_steps=FLAGS.num_train_steps,
      eval_steps=FLAGS.num_eval_steps,
      continuous_eval_throttle_secs=FLAGS.continuous_eval_throttle_secs,
      eval_delay_secs=0)
  # To migrate to tf.estimator.RunConfig.
  run_config = learn_runner.EstimatorConfig(
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      save_summary_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)
  learn_runner.run(
      experiment_fn=experiment_fn, run_config=run_config, hparams=hparams)
if __name__ == '__main__':
  flags.mark_flag_as_required('experiment_config')
  tf.app.run()
