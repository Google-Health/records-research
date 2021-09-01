# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment setup for estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


def make_training_spec_fn(model_instance,
                          num_train_steps,
                          num_eval_steps,
                          save_checkpoints_secs,
                          save_checkpoints_steps=None,
                          model_dir_override=None):
  """Creates function to create estimator, train_spec, eval_specs for training.

  Args:
    model_instance: An model instance for model type-specific code.
    num_train_steps: The number of steps to run training for.
    num_eval_steps: The number of steps to run evaluation for.
    save_checkpoints_secs: Minimum time between consecutive checkpoints being
      saved. Overridden by save_checkpoints_steps.
    save_checkpoints_steps: Number of steps between consecutive checkpoints
      being saved. Overrides save_checkpoints_secs.
    model_dir_override: Override for model output. Cannot be used for
      non-local runs.

  Returns:
    estimator: An Estimator instance to train and evaluate.
    train_spec: A TrainSpec instance to specify the training specification.
    eval_specs: A list of EvalSpec instance to specify the evaluation and export
      specification.
  """

  def _training_spec_fn(hparams):
    """Creates a DistributedTrainingSpec.

    Args:
      hparams: Hyperparameter overrides for model.

    Returns:
      A DistributedTrainingSpec for model training and evaluation.
    """
    # model_dir is typically provided to RunConfig via the TF_CONFIG environment
    # variable, set by the Estimator borg templates. Override is available for
    # local runs or testing.
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir_override,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
    )
    # The base directory for logs and model data. For hyperparameter tuning the
    # Estimator should write data into a different subdirectory for each trial.
    # The `VizierTuner` has already handled it by appending the trial_id to the
    # base directory.
    model_dir = run_config.model_dir
    tf.logging.info('Creating experiment, storing model files in %s',
                    model_dir)
    hparams = model_instance.create_hparams(hparams_overrides=hparams)

    def get_input_fn(mode):

      # Here is the major data simplification. Because we can not share the EHR
      # data and all the relevant preprocessing modules directly, here we use
      # simulated data. We chose toward this setting because it can show the
      # end to end training of the model. Here we want to emphasize and open
      # source the multimodal architecture we learned.
      def input_fn():
        # Here we assume all sequences have the same sequence length, which is
        # 10. In the real EHR data, the sequence length could be very different
        # for different patients. We need to do paddings.
        sequence_length = 10
        # Here we assume dimensions [0:8] are for categorical features;
        # Dimensions [8:24] are for continuous feature; Dimensions [24:56] are
        # for the clinical notes. They are for the exemplified purpose.
        hidden_dimension = 56
        if mode == tf.estimator.ModeKeys.TRAIN:
          num_examples = 800
        elif mode == tf.estimator.ModeKeys.EVAL:
          num_examples = 100
        # Randomly generate simulated data.
        features = np.float32(np.random.sample(
            (num_examples, sequence_length, hidden_dimension)))
        # Randomly generate simulated labels.
        labels = np.float32(np.random.randint(2, size=(num_examples, 1)))
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.repeat().batch(hparams.batch_size)

        data_iter = dataset.make_one_shot_iterator()
        return data_iter.get_next()

      return input_fn

    train_input_fn = get_input_fn(tf.estimator.ModeKeys.TRAIN)
    eval_input_fn = get_input_fn(tf.estimator.ModeKeys.EVAL)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=num_train_steps)

    eval_specs = [
        tf.estimator.EvalSpec(
            input_fn=eval_input_fn, steps=num_eval_steps,
            start_delay_secs=30, name='continuous', throttle_secs=10),
    ]

    estimator = tf.estimator.Estimator(
        model_fn=model_instance.create_model_fn(hparams),
        config=run_config)
    return estimator, train_spec, eval_specs

  return _training_spec_fn


def run(model_instance,
        num_train_steps,
        num_eval_steps,
        save_checkpoints_secs,
        save_checkpoints_steps=None):
  """Runs training and evaluation or tunes with Vizier.

  This function is suitable for runs on borg or local one-off runs. There will
  be output files but nothing returned by this function. If return values are
  needed (e.g. model metrics for tests) use make_training_spec_fn() instead.

  Args:
    model_instance: An instance of a subclass of ModelInterface used for model
      type-specific code.
    num_train_steps: The number of steps to run training for.
    num_eval_steps: The number of steps to run evaluation for.
    save_checkpoints_secs: Minimum time between consecutive checkpoints being
      saved. Overridden by save_checkpoints_steps.
    save_checkpoints_steps: Number of steps between consecutive checkpoints
      being saved. Overrides save_checkpoints_secs.
  """
  if save_checkpoints_steps:
    tf.logging.info('Ignoring save_checkpoints_secs since '
                    'save_checkpoints_steps was provided.')
    save_checkpoints_secs = None

  training_spec_fn = make_training_spec_fn(
      model_instance=model_instance,
      num_train_steps=num_train_steps,
      num_eval_steps=num_eval_steps,
      save_checkpoints_secs=save_checkpoints_secs,
      save_checkpoints_steps=save_checkpoints_steps)

  estimator, train_spec, eval_specs = training_spec_fn(None)
  model_eval_metrics, export_results = tf.estimator.train_and_evaluate(
      estimator, train_spec, eval_specs[0])
