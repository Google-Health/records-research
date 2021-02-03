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
"""Main trainer for unsupervised learning of patient representation using meta learning."""

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import logging
import encoder_decoder
import mimic_data_gen
import tasks

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', '',
                    'dataset_name must be either sinusoid or mimic3.')
flags.DEFINE_string('experiment_config', '',
                    'Path to experimental config of mimic3.')
flags.DEFINE_string(
    'mimic_data_path', '',
    'When not empty, path to mimic3 experiment data'
)
flags.DEFINE_string(
    'transfer_data_path', '',
    'Path to the target data set for generalization experiment. If empty string, then the experiment on generalization to a different dataset will not run'
)

flags.DEFINE_string('job_prefix', '',
                    'Job name as a prefix for metric log file.')
flags.DEFINE_string(
    'experiment_log_path',
    '/meta_learn_logs/',
    'Path to experiment log files.')
flags.DEFINE_boolean('export_to_tensorboard', False,
                     'Whether to export logs for tensorboard.')

flags.DEFINE_integer('num_epochs_supervised', 3,
                     'Num of epochs for supervised training.')
flags.DEFINE_integer('num_epochs_unsupervised', 3,
                     'Num of epochs for unsupervised training.')

flags.DEFINE_integer('num_steps_unsupervised', 300,
                     'Num of epochs for self-supervised training.')
flags.DEFINE_integer('num_steps_supervised', 300,
                     'Num of epochs for supervised training.')
flags.DEFINE_integer('num_epochs_outer_attention', 10,
                     'Num of epochs for learning attention.')

flags.DEFINE_integer('train_batch_size', 32, 'Train batch size.')
flags.DEFINE_integer('eval_batch_size', 4000, 'Eval batch size.')

flags.DEFINE_integer('eval_supervised_every_nsteps', 100,
                     'Run eval for supervised learning every n steps.')
flags.DEFINE_integer('eval_unsupervised_every_nsteps', 100,
                     'Run eval for unsupervised learning every n steps.')

flags.DEFINE_integer('log_steps', 100,
                     'Num steps to log metrics under train mode.')
flags.DEFINE_integer(
    'train_data_percentage', 100,
    'The percentage of data used for training. This parameter is used to evaluate the data efficiency'
)

# This flag is used to verify that the model quality from customized training
# loop is consistent with the model.fit one.
flags.DEFINE_boolean(
    'supervised_train_with_keras_model_fit', False,
    'Whether use Keras model.fit to train the model. If false, use customized training loop.'
)
flags.DEFINE_boolean(
    'unsupervised_train_with_keras_model_fit', False,
    'Whether use Keras model.fit to train the model. If false, use customized training loop.'
)

flags.DEFINE_integer(
    'predict_horizon', 10,
    'Prediction window is [trigger_time, trigger_time + predict_horizon)')

flags.DEFINE_integer('run_id', 0, 'ID for multiple experiment runs.')

flags.DEFINE_integer(
    'fold', 0,
    'If positive, to be used with mimic_data_path, which serves as a path template.'
)

flags.DEFINE_float('supervised_learning_rate', 0.001,
                   'Learning rate of the supervised task.')
flags.DEFINE_float('unsupervised_learning_rate', 0.001,
                   'Learning rate of the unsupervised task.')
flags.DEFINE_float('attention_learning_rate', 0.01,
                   'Learning rate of adjusting attention.')

flags.DEFINE_integer('state_size', 10, 'Size of LSTM state.')

flags.DEFINE_string(
    'train_mode', 'supervised',
    'One of supervised, unsupervised, pretrain, maml_pretrain.')

flags.DEFINE_string('encoder_name', 'lstm', 'Name of encoder.')
flags.DEFINE_string('unsupervised_decoder_name', 'lstm',
                    'Name of decoder for unsupervised training.')
flags.DEFINE_string('supervised_decoder_name', 'mlp',
                    'Name of decoder for supervised training.')

flags.DEFINE_integer(
    'target_feature_index', 100,
    'Index for unsupervised feature trajectory forecast selection. '
    'If target_feature_index >= num_feature, all features are used.')

flags.DEFINE_string(
    'feature_indice_included_str', '',
    'Included index for unsupervised feature trajectory forecast.')
flags.DEFINE_string(
    'feature_indice_excluded_str', '',
    'Excluded index for unsupervised feature trajectory forecast.')

flags.DEFINE_boolean('use_mask_feature', False,
                     'Whether to use mask as a feature.')
flags.DEFINE_boolean(
    'forecast_only_nonmask_values', True,
    'Compute loss only using the true/nonpadded value in trajectory forecast.')
flags.DEFINE_string(
    'forecast_loss_func', 'square',
    'Loss function used for trajectory forecast. Either square or abs.')

flags.DEFINE_boolean(
    'use_best_encoder', True,
    'Use the best unsupervised model as the pretrain checkpoint. If false, take the last checkpoint.'
)

flags.DEFINE_boolean('freeze_encoder_after_pretrain', True,
                     'Whether to freeze the encoder after pretraining.')

flags.DEFINE_boolean('use_attention', False,
                     'Whether to use attention under cotrain mode.')
flags.DEFINE_boolean('use_attn_output', False,
                     'Whether to log attention output vs variable.')

flags.DEFINE_string('normalization_model', 'softmax',
                    'Normalization method in attention layer.')

flags.DEFINE_string('attention_init_value', None,
                    'Init attention values in string with comma as separator.')

flags.DEFINE_float('cotrain_classification_loss_factor', 10,
                   'Learning rate of the supervised task.')

flags.DEFINE_boolean('update_metrics_during_training', False,
                     'Whether to update metrics during training.')

# Used as clip value for all optimizers.
_GRADIENT_CLIP_VALUE = 5


def get_num_element(dataset, data_percentage):
  """Get the num of elements in the dataset when data_percentage is used."""
  total_num = sum(1 for _ in dataset)
  return int(total_num * data_percentage / 100)


def get_datasets(dataset_name, data_path):
  """Get train, eval, test datasets and their config from the given dataset_name."""
  if dataset_name == 'mimic3':
    logging.info(data_path)
    train_dataset, eval_dataset, test_dataset, dataset_config = generate_mimic_datasets(
        data_path=data_path,
        experiment_config_path=FLAGS.experiment_config,
        predict_horizon=FLAGS.predict_horizon)
  else:
    raise ValueError('dataset_name must be mimic3.')
  return train_dataset, eval_dataset, test_dataset, dataset_config


def generate_mimic_datasets(data_path, experiment_config_path, predict_horizon):
  """Generate tf.dataset(s) with MIMIC3 data at given data_path.

  Args:
    data_path: Path to TF examples from mimic3 dataset.
    experiment_config_path: Path to the experiment config for parse the TF
      Example features.
    predict_horizon: The length of time units for self-supervised prediction
      task labels.

  Returns:
    Three tf.datasets for train, eval, test and a dataset_config.
  """
  gen = mimic_data_gen.MIMIC3DataGenerator.from_config(
      experiment_config=experiment_config_path, predict_horizon=predict_horizon)
  train_dataset = gen.get_dataset(data_path=data_path, mode='train')
  eval_dataset = gen.get_dataset(data_path=data_path, mode='eval')
  test_dataset = gen.get_dataset(data_path=data_path, mode='test')

  dataset_config = gen.get_config()

  return train_dataset, eval_dataset, test_dataset, dataset_config


def _flatten_gradient(grad_list):
  flatten_gradient = tf.concat([tf.reshape(var, [-1]) for var in grad_list],
                               axis=0)
  return flatten_gradient


def _flatten_hessian(hessian_list):
  flattened_hessians = tf.concat(
      [tf.reshape(hess, [hess.shape[0], -1]) for hess in hessian_list], 1)

  return flattened_hessians


def _apply_gradients(optimizer, gradients, variables):
  optimizer.apply_gradients(zip(gradients, variables))


def compute_loss(feature,
                 label,
                 encoder,
                 decoder,
                 task,
                 mode,
                 step,
                 reset_metrics=False,
                 update_metrics=True):
  """Compute loss and update the task metrics."""
  feature_value, feature_mask = feature
  if FLAGS.use_mask_feature:
    feature_mask = tf.cast(feature_mask, dtype=feature_value.dtype)
    feature_value_with_mask = tf.concat([feature_value, feature_mask], axis=2)
    output, state = encoder.call(feature_value_with_mask)
  else:
    output, state = encoder.call(feature_value)
  prediction = decoder.call((output, state))

  loss = task.get_loss(label, prediction, mode)
  metrics = None
  if update_metrics:
    metrics = task.update_metrics(label, prediction, mode, step, reset_metrics)

  return state, prediction, loss, metrics


def train_batch_inner_unsupervised_hessian(batch_index, train_feature_batch,
                                           train_label_batch, encoder,
                                           unsupervised_decoder,
                                           regression_task, optimizer,
                                           global_train_step):
  """One self-supervised training step in meta learning with Hessian."""
  with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
      _, _, unsupervised_loss, _ = compute_loss(
          train_feature_batch,
          train_label_batch,
          encoder,
          unsupervised_decoder,
          regression_task,
          'train',
          global_train_step,
          reset_metrics=batch_index == 0,
          update_metrics=FLAGS.update_metrics_during_training)
    encoder_grad = inner_tape.gradient(unsupervised_loss,
                                       encoder.trainable_variables)
    decoder_grad = inner_tape.gradient(unsupervised_loss,
                                       unsupervised_decoder.trainable_variables)
    _apply_gradients(optimizer, encoder_grad, encoder.trainable_variables)
    _apply_gradients(optimizer, decoder_grad,
                     unsupervised_decoder.trainable_variables)

    # with state_size = 10, encoder_grad (same as encoder.trainable_variables)
    # is a list of three tensors with shape (96, 40), (10, 40), (40,).
    # After flattenning, shape=(4280,)
    # Note that tf.hessian. tf.gradients is not supported when eager execution
    # is enabled.
    flattened_encoder_grad = _flatten_gradient(encoder_grad)

  # Note outer_tape.gradient only provides the diag vector of the hessian.
  # Need to use outer_tape.jacobian here.
  # hessian_encoder is a list of three tensors with shape
  # (4280, 96, 40),  (4280, 10, 40), (4280, 40),
  hessian_encoder = outer_tape.jacobian(flattened_encoder_grad,
                                        encoder.trainable_variables)

  flattened_hessians = _flatten_hessian(hessian_encoder)

  # j_encoder is a list of one tensors [(4280, 96)]
  j_encoder = outer_tape.jacobian(flattened_encoder_grad,
                                  regression_task.attention_variables)
  flattened_j_encoder = j_encoder[0]

  del inner_tape
  del outer_tape

  return flattened_hessians, flattened_j_encoder


def train_batch_inner_supervised_hessian(batch_index, train_feature_batch,
                                         train_label_batch, encoder,
                                         supervised_decoder,
                                         classification_task, optimizer,
                                         global_train_step):
  """One supervised training step in meta learning with Hessian."""
  with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
      _, _, supervised_loss, _ = compute_loss(
          train_feature_batch,
          train_label_batch,
          encoder,
          supervised_decoder,
          classification_task,
          'train',
          global_train_step,
          reset_metrics=batch_index == 0,
          update_metrics=FLAGS.update_metrics_during_training)

    encoder_grad = inner_tape.gradient(supervised_loss,
                                       encoder.trainable_variables)
    decoder_grad = inner_tape.gradient(supervised_loss,
                                       supervised_decoder.trainable_variables)
    _apply_gradients(optimizer, encoder_grad, encoder.trainable_variables)
    _apply_gradients(optimizer, decoder_grad,
                     supervised_decoder.trainable_variables)
    flattened_encoder_grad = _flatten_gradient(encoder_grad)

  hessian_encoder = outer_tape.jacobian(flattened_encoder_grad,
                                        encoder.trainable_variables)
  flattened_hessians = _flatten_hessian(hessian_encoder)

  del inner_tape
  del outer_tape

  return flattened_hessians, encoder_grad


def train_batch_inner_unsupervised_approx(batch_index, train_feature_batch,
                                          train_label_batch, encoder,
                                          unsupervised_decoder, regression_task,
                                          optimizer, global_train_step):
  """One unsupervised training step in meta learning with approximation."""
  with tf.GradientTape(persistent=True) as inner_tape:
    _, _, unsupervised_loss, _ = compute_loss(
        train_feature_batch,
        train_label_batch,
        encoder,
        unsupervised_decoder,
        regression_task,
        'train',
        global_train_step,
        reset_metrics=batch_index == 0)
  innner_grad = inner_tape.gradient(
      unsupervised_loss,
      encoder.trainable_variables + unsupervised_decoder.trainable_variables)
  _apply_gradients(
      optimizer, innner_grad,
      encoder.trainable_variables + unsupervised_decoder.trainable_variables)
  dLu_dphi = inner_tape.gradient(unsupervised_loss, encoder.trainable_variables)  # pylint: disable=invalid-name
  dLu_dlambda = inner_tape.gradient(  # pylint: disable=invalid-name
      unsupervised_loss, regression_task.attention_variables)
  del inner_tape
  return dLu_dphi, dLu_dlambda


def train_batch_inner_supervised_approx(batch_index, train_feature_batch,
                                        train_label_batch, eval_dataset,
                                        encoder, supervised_decoder,
                                        classification_task, optimizer,
                                        global_train_step):
  """One supervised training step in meta learning with approximation."""

  with tf.GradientTape(persistent=True) as inner_tape_train:
    # Loss on train dataset for gradient update to simulate fine tuning of
    # classification head.
    _, _, supervised_train_loss, _ = compute_loss(
        train_feature_batch,
        train_label_batch,
        encoder,
        supervised_decoder,
        classification_task,
        'train',
        global_train_step,
        reset_metrics=batch_index == 0,
        update_metrics=False)

  # fine tuning classification head.
  supervised_head_grad = inner_tape_train.gradient(
      supervised_train_loss, supervised_decoder.trainable_variables)
  _apply_gradients(optimizer, supervised_head_grad,
                   supervised_decoder.trainable_variables)

  with tf.GradientTape(persistent=True) as inner_tape_eval:
    # Only one step as eval_batch_size is the same/larger than eval datasize.
    for (eval_feature_batch, eval_supervised_label_batch,
         eval_unsupervised_label_batch) in eval_dataset:
      eval_label_batch = (eval_supervised_label_batch,
                          eval_unsupervised_label_batch)
      _, _, supervised_eval_loss, _ = compute_loss(
          eval_feature_batch,
          eval_label_batch,
          encoder,
          supervised_decoder,
          classification_task,
          'train',
          global_train_step,
          reset_metrics=False,
          update_metrics=False)

  # Compute hyper-gradient.
  dLs_dphi = inner_tape_eval.gradient(  # pylint: disable=invalid-name
      supervised_eval_loss, encoder.trainable_variables)
  del inner_tape_eval
  del inner_tape_train
  return dLs_dphi


def train_batch_cotrain(
    unsupervised_train_feature_batch, unsupervised_train_label_batch,
    supervised_train_feature_batch, supervised_train_supervised_label_batch,
    encoder, unsupervised_decoder, supervised_decoder, regression_task,
    classification_task, unsupervised_optimizer, supervised_optimizer, step):
  """One training step."""
  with tf.GradientTape(persistent=True) as tape:
    _, _, regression_loss, _ = compute_loss(unsupervised_train_feature_batch,
                                            unsupervised_train_label_batch,
                                            encoder, unsupervised_decoder,
                                            regression_task, 'train', step)
    _, _, classification_loss, _ = compute_loss(
        supervised_train_feature_batch, supervised_train_supervised_label_batch,
        encoder, supervised_decoder, classification_task, 'train', step)

    aggregated_loss = regression_loss + (
        FLAGS.cotrain_classification_loss_factor * classification_loss)

  encoder_gradients = tape.gradient(aggregated_loss,
                                    encoder.trainable_variables)
  unsupervised_decoder_gradients = tape.gradient(
      aggregated_loss, unsupervised_decoder.trainable_variables)
  supervised_decoder_gradients = tape.gradient(
      aggregated_loss, supervised_decoder.trainable_variables)

  _apply_gradients(unsupervised_optimizer, encoder_gradients,
                   encoder.trainable_variables)
  _apply_gradients(unsupervised_optimizer, unsupervised_decoder_gradients,
                   unsupervised_decoder.trainable_variables)
  _apply_gradients(supervised_optimizer, supervised_decoder_gradients,
                   supervised_decoder.trainable_variables)

  del tape


def train_batch(feature, label, encoder, decoder, task, optimizer, step,
                freeze_encoder):
  """One training step."""
  with tf.GradientTape(persistent=True) as tape:
    _, _, loss, _ = compute_loss(feature, label, encoder, decoder, task,
                                 'train', step)
  encoder_gradients = tape.gradient(loss, encoder.trainable_variables)
  decoder_gradients = tape.gradient(loss, decoder.trainable_variables)

  if freeze_encoder:
    _apply_gradients(optimizer, decoder_gradients, decoder.trainable_variables)
  else:
    _apply_gradients(optimizer, encoder_gradients + decoder_gradients,
                     encoder.trainable_variables + decoder.trainable_variables)
  del tape


def eval_all(eval_dataset, encoder, decoder, task, mode, train_step):
  """Eval the entire eval_dataset."""
  for eval_step, (eval_feature_batch, eval_supervised_label_batch,
                  eval_unsupervised_label_batch) in enumerate(eval_dataset):
    eval_label_batch = (eval_supervised_label_batch,
                        eval_unsupervised_label_batch)
    state, prediction, loss, metrics = compute_loss(
        eval_feature_batch,
        eval_label_batch,
        encoder,
        decoder,
        task,
        mode,
        train_step + eval_step,
        reset_metrics=eval_step == 0)
  return state, prediction, loss, metrics


def continuous_eval(eval_dataset, test_dataset, train_feature_batch, encoder,
                    best_encoder, unsupervised_decoder, supervised_decoder,
                    regression_task, classification_task, best_metrics,
                    global_train_step):
  """Check and run eval at global_train_step."""
  if global_train_step % FLAGS.eval_unsupervised_every_nsteps == 0:
    eval_all(eval_dataset, encoder, unsupervised_decoder, regression_task,
             'eval', global_train_step)
  if global_train_step % FLAGS.eval_supervised_every_nsteps == 0:
    _, _, _, metrics = eval_all(eval_dataset, encoder, supervised_decoder,
                                classification_task, 'eval', global_train_step)
    auc, _ = metrics
    if auc > best_metrics:
      best_metrics = auc
      best_encoder = copy_encoder(encoder, train_feature_batch)
      eval_all(test_dataset, encoder, supervised_decoder, classification_task,
               'test', global_train_step)
  return best_metrics, best_encoder


class EncoderDecoderModel(tf.keras.Model):
  """A keras model with encoder/decoder architecture."""

  def __init__(self, encoder, decoder, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder

  def call(self, inputs):
    feature_value, feature_mask = inputs
    if FLAGS.use_mask_feature:
      feature_mask = tf.cast(feature_mask, dtype=tf.float32)
      feature_value_with_mask = tf.concat([feature_value, feature_mask], axis=2)
      output, state = self.encoder(feature_value_with_mask)
    else:
      output, state = self.encoder(feature_value)
    prediction = self.decoder((output, state))
    return prediction

  def get_encoder(self):
    return self.encoder


def copy_encoder(encoder, x):
  """Copy model weights to a new model.

  Args:
    encoder: encoder model to be copied.
    x: An input example. This is used to run a forward pass in order to add the
      weights of the graph as variables.

  Returns:
      A copy of the model.
  """
  encoder_model = ENCODER_MAP[FLAGS.encoder_name]
  copied_encoder = encoder_model(state_size=FLAGS.state_size)

  # If we don't run this step the weights are not "initialized"
  # and the gradients will not be computed.
  feature_value, feature_mask = x

  if FLAGS.use_mask_feature:
    feature_mask = tf.cast(feature_mask, dtype=tf.float32)
    feature_value_with_mask = tf.concat([feature_value, feature_mask], axis=2)
    _, _ = copied_encoder.call(feature_value_with_mask)
  else:
    copied_encoder.call(feature_value)
  copied_encoder.set_weights(encoder.get_weights())
  return copied_encoder


def train_supervising_unsupervised_model_attn_hessian(train_dataset,
                                                      eval_dataset,
                                                      test_dataset,
                                                      dataset_config):
  """Targeted unsupervised learning to optimize given supervised objective."""
  # In this method, the hessian matrix is implemented directly.
  feature_keys, num_predict = dataset_config
  num_feature = len(feature_keys)

  encoder_model = ENCODER_MAP[FLAGS.encoder_name]
  print('create new encoder: ' + FLAGS.encoder_name)
  encoder = encoder_model(state_size=FLAGS.state_size)

  unsupervised_decoder_model = DECODER_MAP[FLAGS.unsupervised_decoder_name]
  print('create new decoder: ' + FLAGS.unsupervised_decoder_name)
  unsupervised_decoder = unsupervised_decoder_model(
      num_feature=num_feature,
      num_predict=num_predict,
      state_size=FLAGS.state_size)

  supervised_decoder_model = DECODER_MAP[FLAGS.supervised_decoder_name]
  print('create new decoder: ' + FLAGS.supervised_decoder_name)
  supervised_decoder = supervised_decoder_model()

  train_dataset = train_dataset.batch(FLAGS.train_batch_size)
  eval_dataset = eval_dataset.batch(FLAGS.eval_batch_size)
  test_dataset = test_dataset.batch(FLAGS.eval_batch_size)

  unsupervised_optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.unsupervised_learning_rate,
      clipvalue=_GRADIENT_CLIP_VALUE)
  supervised_optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.supervised_learning_rate,
      clipvalue=_GRADIENT_CLIP_VALUE)
  attention_optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.attention_learning_rate,
      clipvalue=_GRADIENT_CLIP_VALUE)

  if FLAGS.unsupervised_train_with_keras_model_fit:
    raise ValueError(
        'unsupervised_train_with_keras_model_fit not supported under attn_hessian'
    )

  regression_task = tasks.RegressionTask(
      num_feature, num_predict, FLAGS.target_feature_index,
      FLAGS.feature_indice_included_str, FLAGS.feature_indice_excluded_str,
      feature_keys, FLAGS.log_steps, FLAGS.export_to_tensorboard,
      FLAGS.experiment_log_path, FLAGS.job_prefix,
      FLAGS.forecast_only_nonmask_values, FLAGS.forecast_loss_func,
      FLAGS.use_attention, FLAGS.normalization_model,
      FLAGS.attention_init_value, FLAGS.use_attn_output)

  classification_task = tasks.ClassificationTask(FLAGS.log_steps,
                                                 FLAGS.export_to_tensorboard,
                                                 FLAGS.experiment_log_path,
                                                 FLAGS.job_prefix)
  global_train_step = 0
  best_metrics = 0
  best_encoder = encoder
  num_batch_supervised_train_dataset = get_num_element(
      train_dataset, FLAGS.train_data_percentage)
  num_batch_unsupervised_train_dataset = get_num_element(train_dataset, 100)

  supervised_train_dataset = train_dataset.take(
      num_batch_supervised_train_dataset)
  unsupervised_iter = iter(train_dataset)
  supervised_iter = iter(supervised_train_dataset)
  unsupervised_batch_index = 0
  supervised_batch_index = 0

  for _ in range(FLAGS.num_epochs_outer_attention):
    hessian_encoder_list = []
    j_encoder_list = []

    for _ in range(FLAGS.num_steps_unsupervised):
      (unsupervised_train_feature_batch,
       unsupervised_train_supervised_label_batch,
       unsupervised_train_unsupervised_label_batch) = next(unsupervised_iter)
      unsupervised_batch_index += 1
      unsupervised_train_label_batch = (
          unsupervised_train_supervised_label_batch,
          unsupervised_train_unsupervised_label_batch)
      hessian_encoder, j_encoder = train_batch_inner_unsupervised_hessian(
          unsupervised_batch_index, unsupervised_train_feature_batch,
          unsupervised_train_label_batch, encoder, unsupervised_decoder,
          regression_task, unsupervised_optimizer, global_train_step)

      if unsupervised_batch_index >= num_batch_unsupervised_train_dataset:
        unsupervised_iter = iter(train_dataset)
        unsupervised_batch_index = 0
      # hessian_encoder_reshape shape [num_encoder_weights, num_encoder_weights]
      hessian_encoder_list.append(FLAGS.unsupervised_learning_rate *
                                  hessian_encoder)
      # j_encoder matrix with shape num_attention_weights * num_encoder_weights.
      j_encoder_list.append(FLAGS.unsupervised_learning_rate * j_encoder)
      global_train_step += 1

    for _ in range(FLAGS.num_steps_supervised):
      (supervised_train_feature_batch, supervised_train_supervised_label_batch,
       supervised_train_unsupervised_label_batch) = next(supervised_iter)
      supervised_batch_index += 1
      supervised_train_label_batch = (supervised_train_supervised_label_batch,
                                      supervised_train_unsupervised_label_batch)
      hessian_encoder, encoder_grad = train_batch_inner_supervised_hessian(
          supervised_batch_index, supervised_train_feature_batch,
          supervised_train_label_batch, encoder, supervised_decoder,
          classification_task, supervised_optimizer, global_train_step)
      if supervised_batch_index >= num_batch_supervised_train_dataset:
        supervised_iter = iter(supervised_train_dataset)
        supervised_batch_index = 0
      hessian_encoder_list.append(FLAGS.supervised_learning_rate *
                                  hessian_encoder)
    # list with len num_encoder_weights (4280,)
    beta = encoder_grad
    # beta_reshape shape=(1, 4280)
    beta_reshape = tf.expand_dims(_flatten_gradient(beta), axis=0)

    for index in range(FLAGS.num_steps_supervised):
      # hessian_encoder [num_encoder_weights, num_encoder_weights]
      beta_reshape = tf.matmul(beta_reshape,
                               (1 - hessian_encoder_list[-(index + 1)]))
    alpha = beta_reshape
    attention_grad = tf.Variable(tf.zeros_initializer()(
        shape=[num_feature], dtype=tf.float32))

    for index in range(FLAGS.num_steps_unsupervised):
      attention_grad = attention_grad - (
          tf.matmul(alpha, j_encoder_list[-(index + 1)]))
      hessian_encoder_reshape = hessian_encoder_list[-(
          FLAGS.num_steps_supervised + index + 1)]
      alpha = tf.matmul(alpha, (1 - hessian_encoder_reshape))
    _apply_gradients(attention_optimizer, attention_grad,
                     regression_task.attention_variables)

    # continuous eval.
    best_metrics, best_encoder = continuous_eval(
        eval_dataset, test_dataset, unsupervised_train_feature_batch, encoder,
        best_encoder, unsupervised_decoder, supervised_decoder, regression_task,
        classification_task, best_metrics, global_train_step)

  return encoder, best_encoder, global_train_step


def train_supervising_unsupervised_model_attn_approx(train_dataset,
                                                     train_data_percentage,
                                                     eval_dataset, test_dataset,
                                                     dataset_config):
  """Train a baseline supervised model based on binary label."""
  feature_keys, num_predict = dataset_config
  num_feature = len(feature_keys)

  encoder_model = ENCODER_MAP[FLAGS.encoder_name]
  print('create new encoder: ' + FLAGS.encoder_name)
  encoder = encoder_model(state_size=FLAGS.state_size)

  unsupervised_decoder_model = DECODER_MAP[FLAGS.unsupervised_decoder_name]
  print('create new decoder: ' + FLAGS.unsupervised_decoder_name)
  unsupervised_decoder = unsupervised_decoder_model(
      num_feature=num_feature,
      num_predict=num_predict,
      state_size=FLAGS.state_size)

  supervised_decoder_model = DECODER_MAP[FLAGS.supervised_decoder_name]
  print('create new decoder: ' + FLAGS.supervised_decoder_name)
  supervised_decoder = supervised_decoder_model()

  unsupervised_optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.unsupervised_learning_rate,
      clipvalue=_GRADIENT_CLIP_VALUE)
  supervised_optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.supervised_learning_rate,
      clipvalue=_GRADIENT_CLIP_VALUE)
  attention_optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.attention_learning_rate,
      clipvalue=_GRADIENT_CLIP_VALUE)

  if FLAGS.unsupervised_train_with_keras_model_fit:
    raise ValueError(
        'unsupervised_train_with_keras_model_fit not supported under attn_approx'
    )

  regression_task = tasks.RegressionTask(
      num_feature, num_predict, FLAGS.target_feature_index,
      FLAGS.feature_indice_included_str, FLAGS.feature_indice_excluded_str,
      feature_keys, FLAGS.log_steps, FLAGS.export_to_tensorboard,
      FLAGS.experiment_log_path, FLAGS.job_prefix,
      FLAGS.forecast_only_nonmask_values, FLAGS.forecast_loss_func,
      FLAGS.use_attention, FLAGS.normalization_model,
      FLAGS.attention_init_value, FLAGS.use_attn_output)

  classification_task = tasks.ClassificationTask(FLAGS.log_steps,
                                                 FLAGS.export_to_tensorboard,
                                                 FLAGS.experiment_log_path,
                                                 FLAGS.job_prefix)

  train_dataset = train_dataset.batch(FLAGS.train_batch_size)
  eval_dataset = eval_dataset.batch(FLAGS.eval_batch_size)
  test_dataset = test_dataset.batch(FLAGS.eval_batch_size)

  num_batch_supervised_train_dataset = get_num_element(train_dataset,
                                                       train_data_percentage)
  num_batch_unsupervised_train_dataset = get_num_element(train_dataset, 100)
  supervised_train_dataset = train_dataset.take(
      num_batch_supervised_train_dataset)

  unsupervised_iter = iter(train_dataset)
  supervised_iter = iter(supervised_train_dataset)

  unsupervised_batch_index = 0
  supervised_batch_index = 0

  global_train_step = 0
  best_metrics = 0
  best_encoder = encoder
  for _ in range(FLAGS.num_epochs_outer_attention):
    for _ in range(FLAGS.num_steps_unsupervised):
      (unsupervised_train_feature_batch,
       unsupervised_train_supervised_label_batch,
       unsupervised_train_unsupervised_label_batch) = next(unsupervised_iter)
      unsupervised_batch_index += 1
      unsupervised_train_label_batch = (
          unsupervised_train_supervised_label_batch,
          unsupervised_train_unsupervised_label_batch)

      dLu_dphi, dLu_dlambda = train_batch_inner_unsupervised_approx(  # pylint: disable=invalid-name
          unsupervised_batch_index, unsupervised_train_feature_batch,
          unsupervised_train_label_batch, encoder, unsupervised_decoder,
          regression_task, unsupervised_optimizer, global_train_step)
      if unsupervised_batch_index >= num_batch_unsupervised_train_dataset:
        unsupervised_iter = iter(train_dataset)
        unsupervised_batch_index = 0
      global_train_step += 1

      best_metrics, best_encoder = continuous_eval(
          eval_dataset, test_dataset, unsupervised_train_feature_batch, encoder,
          best_encoder, unsupervised_decoder, supervised_decoder,
          regression_task, classification_task, best_metrics, global_train_step)

    for _ in range(FLAGS.num_steps_supervised):
      (supervised_train_feature_batch, supervised_train_supervised_label_batch,
       supervised_train_unsupervised_label_batch) = next(supervised_iter)
      supervised_batch_index += 1
      supervised_train_label_batch = (supervised_train_supervised_label_batch,
                                      supervised_train_unsupervised_label_batch)

      dLs_dphi = train_batch_inner_supervised_approx(  # pylint: disable=invalid-name
          supervised_batch_index, supervised_train_feature_batch,
          supervised_train_label_batch, eval_dataset, encoder,
          supervised_decoder, classification_task, supervised_optimizer,
          global_train_step)
      if supervised_batch_index >= num_batch_supervised_train_dataset:
        supervised_iter = iter(supervised_train_dataset)
        supervised_batch_index = 0
      global_train_step += 1

      best_metrics, best_encoder = continuous_eval(
          eval_dataset, test_dataset, unsupervised_train_feature_batch, encoder,
          best_encoder, unsupervised_decoder, supervised_decoder,
          regression_task, classification_task, best_metrics, global_train_step)

    if FLAGS.use_attention:
      # dLs_dlambda = dLs_dphi * dphi_dlambda
      # dphi_dlambda = dphi_dLu * dLu_dlambda = dLu_dphi^-1 * dLu_dlambda

      dLs_dphi_reshape = _flatten_gradient(dLs_dphi)  # pylint: disable=invalid-name
      dLu_dphi_reshape = _flatten_gradient(dLu_dphi)  # pylint: disable=invalid-name
      dLu_dlambda_reshape = _flatten_gradient(dLu_dlambda)  # pylint: disable=invalid-name

      dphi_dlambda = tf.matmul(
          tf.expand_dims(tf.math.reciprocal(dLu_dphi_reshape), axis=1),
          tf.expand_dims(dLu_dlambda_reshape, axis=0))  # [phi, lambda]
      dphi_dlambda = tf.clip_by_value(dphi_dlambda, -10000, 10000, name=None)

      attention_grad_tensor = tf.squeeze(
          tf.matmul(tf.expand_dims(dLs_dphi_reshape, axis=0), dphi_dlambda),
          axis=0)  # [lambda]
      attention_grad_tensor = tf.clip_by_value(
          attention_grad_tensor, -10000, 10000, name=None)

      _apply_gradients(attention_optimizer, [attention_grad_tensor.numpy()],
                       regression_task.attention_variables)
  return encoder, best_encoder, global_train_step


def train_cotrain_unsupervised_model(train_dataset, train_data_percentage,
                                     eval_dataset, test_dataset,
                                     dataset_config):
  """Train a baseline supervised model based on binary label."""
  feature_keys, num_predict = dataset_config
  num_feature = len(feature_keys)

  encoder_model = ENCODER_MAP[FLAGS.encoder_name]
  print('create new encoder: ' + FLAGS.encoder_name)
  encoder = encoder_model(state_size=FLAGS.state_size)

  unsupervised_decoder_model = DECODER_MAP[FLAGS.unsupervised_decoder_name]
  print('create new decoder: ' + FLAGS.unsupervised_decoder_name)
  unsupervised_decoder = unsupervised_decoder_model(
      num_feature=num_feature,
      num_predict=num_predict,
      state_size=FLAGS.state_size)

  supervised_decoder_model = DECODER_MAP[FLAGS.supervised_decoder_name]
  print('create new decoder: ' + FLAGS.supervised_decoder_name)
  supervised_decoder = supervised_decoder_model()

  unsupervised_optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.unsupervised_learning_rate,
      clipvalue=_GRADIENT_CLIP_VALUE)
  supervised_optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.supervised_learning_rate,
      clipvalue=_GRADIENT_CLIP_VALUE)

  if FLAGS.unsupervised_train_with_keras_model_fit:
    raise ValueError(
        'unsupervised_train_with_keras_model_fit not supported under cotrain_pretrain'
    )

  regression_task = tasks.RegressionTask(
      num_feature=num_feature,
      num_predict=num_predict,
      target_feature_index=FLAGS.target_feature_index,
      feature_indice_included_str=FLAGS.feature_indice_included_str,
      feature_indice_excluded_str=FLAGS.feature_indice_excluded_str,
      feature_keys=feature_keys,
      log_steps=FLAGS.log_steps,
      export_to_tensorboard=FLAGS.export_to_tensorboard,
      experiment_log_path=FLAGS.experiment_log_path,
      job_prefix=FLAGS.job_prefix,
      forecast_only_nonmask_values=FLAGS.forecast_only_nonmask_values,
      forecast_loss_func=FLAGS.forecast_loss_func,
      use_attention=FLAGS.use_attention,
      normalization_model=FLAGS.normalization_model,
      attention_init_value=FLAGS.attention_init_value,
      use_attn_output=FLAGS.use_attn_output)

  classification_task = tasks.ClassificationTask(FLAGS.log_steps,
                                                 FLAGS.export_to_tensorboard,
                                                 FLAGS.experiment_log_path,
                                                 FLAGS.job_prefix)

  train_dataset = train_dataset.batch(FLAGS.train_batch_size)
  eval_dataset = eval_dataset.batch(FLAGS.eval_batch_size)
  test_dataset = test_dataset.batch(FLAGS.eval_batch_size)

  num_batch_supervised_train_dataset = get_num_element(train_dataset,
                                                       train_data_percentage)
  supervised_train_dataset = train_dataset.take(
      num_batch_supervised_train_dataset)
  supervised_iter = iter(supervised_train_dataset)
  supervised_batch_index = 0

  global_train_step = 0
  best_metrics = 0
  best_encoder = encoder

  for _ in range(FLAGS.num_epochs_unsupervised):
    for (unsupervised_train_feature_batch,
         unsupervised_train_supervised_label_batch,
         unsupervised_train_unsupervised_label_batch) in train_dataset:
      global_train_step += 1
      unsupervised_train_label_batch = (
          unsupervised_train_supervised_label_batch,
          unsupervised_train_unsupervised_label_batch)
      (supervised_train_feature_batch, supervised_train_supervised_label_batch,
       supervised_train_unsupervised_label_batch) = next(supervised_iter)
      supervised_batch_index += 1
      supervised_train_label_batch = (supervised_train_supervised_label_batch,
                                      supervised_train_unsupervised_label_batch)

      if supervised_batch_index >= num_batch_supervised_train_dataset:
        supervised_iter = iter(supervised_train_dataset)
        supervised_batch_index = 0

      train_batch_cotrain(
          unsupervised_train_feature_batch, unsupervised_train_label_batch,
          supervised_train_feature_batch, supervised_train_label_batch, encoder,
          unsupervised_decoder, supervised_decoder, regression_task,
          classification_task, unsupervised_optimizer, supervised_optimizer,
          global_train_step)

      best_metrics, best_encoder = continuous_eval(
          eval_dataset, test_dataset, supervised_train_feature_batch, encoder,
          best_encoder, unsupervised_decoder, supervised_decoder,
          regression_task, classification_task, best_metrics, global_train_step)

  return encoder, best_encoder, global_train_step


def train_unsupervised_model(train_dataset, eval_dataset, dataset_config):
  """Train a baseline supervised model based on binary label."""
  feature_keys, num_predict = dataset_config
  for i, f in enumerate(feature_keys):
    logging.info('%d: %s', i, f)
  num_feature = len(feature_keys)

  encoder_model = ENCODER_MAP[FLAGS.encoder_name]
  logging.info('create new encoder: %s', FLAGS.encoder_name)
  encoder = encoder_model(state_size=FLAGS.state_size)

  decoder_model = DECODER_MAP[FLAGS.unsupervised_decoder_name]
  logging.info('create new decoder: %s', FLAGS.unsupervised_decoder_name)
  decoder = decoder_model(
      num_feature=num_feature,
      num_predict=num_predict,
      state_size=FLAGS.state_size)

  train_dataset = train_dataset.batch(FLAGS.train_batch_size)
  eval_dataset = eval_dataset.batch(FLAGS.eval_batch_size)

  optimizer = tf.keras.optimizers.Adam(
      learning_rate=FLAGS.unsupervised_learning_rate)

  ### using keras.Model and model fit.
  if FLAGS.unsupervised_train_with_keras_model_fit:
    model = EncoderDecoderModel(encoder, decoder)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()])
    if FLAGS.forecast_only_nonmask_values:
      raise ValueError(
          'forecast_only_nonmask_values is not supported under unsupervised_train_with_keras_model_fit.'
      )

    train_dataset = train_dataset.map(lambda f, sl, ul: (f, ul[0]))
    eval_dataset = eval_dataset.map(lambda f, sl, ul: (f, ul[0]))

    logging.info('# Fit model on training data')
    model.fit(
        train_dataset,
        epochs=FLAGS.num_epochs_unsupervised,
        validation_data=eval_dataset)

    if FLAGS.use_best_encoder:
      raise ValueError(
          'use_best_encoder is not supported under unsupervised_train_with_keras_model_fit.'
      )
    return model.get_encoder(), model.get_encoder(), 0
  else:
    ### Customized Training Loop.
    task = tasks.RegressionTask(
        num_feature,
        num_predict,
        FLAGS.target_feature_index,
        FLAGS.feature_indice_included_str,
        FLAGS.feature_indice_excluded_str,
        feature_keys,
        FLAGS.log_steps,
        FLAGS.export_to_tensorboard,
        FLAGS.experiment_log_path,
        FLAGS.job_prefix,
        FLAGS.forecast_only_nonmask_values,
        FLAGS.forecast_loss_func,
        use_attention=False,
        normalization_model=FLAGS.normalization_model,
        attention_init_value=FLAGS.attention_init_value,
        use_attn_output=FLAGS.use_attn_output)
    train_step = 0
    best_metrics = np.inf

    for _ in range(FLAGS.num_epochs_unsupervised):
      for (train_feature_batch, train_supervised_label_batch,
           train_unsupervised_label_batch) in train_dataset:
        train_label_batch = (train_supervised_label_batch,
                             train_unsupervised_label_batch)
        train_batch(train_feature_batch, train_label_batch, encoder, decoder,
                    task, optimizer, train_step, False)
        train_step += 1
        # continuous eval.
        if train_step % FLAGS.eval_unsupervised_every_nsteps == 0:
          _, _, _, metrics = eval_all(eval_dataset, encoder, decoder, task,
                                      'eval', train_step)
          mse, _ = metrics
          if mse < best_metrics:
            best_metrics = mse
            best_encoder = copy_encoder(encoder, train_feature_batch)

    return encoder, best_encoder, train_step


def train_supervised_model(train_dataset,
                           train_data_percentage,
                           eval_dataset,
                           test_dataset,
                           dataset_config,
                           pretrained_encoder=None,
                           freeze_pretrained_encoder=True,
                           init_train_step=0):
  """Train a baseline supervised model based on binary label."""
  del dataset_config
  if pretrained_encoder is None:
    encoder_model = ENCODER_MAP[FLAGS.encoder_name]
    logging.info('create new encoder: %s', FLAGS.encoder_name)
    encoder = encoder_model(state_size=FLAGS.state_size)
  else:
    logging.info('load pretrained encoder')
    encoder = pretrained_encoder
  lr = FLAGS.supervised_learning_rate

  decoder_model = DECODER_MAP[FLAGS.supervised_decoder_name]
  logging.info('create new decoder: %s', FLAGS.supervised_decoder_name)
  decoder = decoder_model()

  num_train_dataset = get_num_element(train_dataset, train_data_percentage)
  train_dataset = train_dataset.take(num_train_dataset)
  train_dataset.shuffle(1000)

  train_dataset = train_dataset.batch(FLAGS.train_batch_size)
  eval_dataset = eval_dataset.batch(FLAGS.eval_batch_size)
  test_dataset = test_dataset.batch(FLAGS.eval_batch_size)

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  ### using keras.Model and model fit.
  if FLAGS.supervised_train_with_keras_model_fit:
    if freeze_pretrained_encoder:
      raise ValueError(
          'freeze_pretrained_encoder is not supported under unsupervised_train_with_keras_model_fit.'
      )

    model = EncoderDecoderModel(encoder, decoder)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False),  # Note this needs to be False.
        metrics=[tf.keras.metrics.AUC(),
                 tf.keras.metrics.AUC(curve='PR')])
    logging.info('# Fit model on training data')
    train_dataset = train_dataset.map(lambda f, sl, ul: (f, sl))
    eval_dataset = eval_dataset.map(lambda f, sl, ul: (f, sl))
    model.fit(
        train_dataset,
        epochs=FLAGS.num_epochs_supervised,
        validation_data=eval_dataset)
  else:
    ### Customized Training Loop.

    task = tasks.ClassificationTask(FLAGS.log_steps,
                                    FLAGS.export_to_tensorboard,
                                    FLAGS.experiment_log_path, FLAGS.job_prefix)
    train_step = init_train_step
    best_metrics = 0

    for epoch in range(FLAGS.num_epochs_supervised):
      for batch_index, (
          train_feature_batch, train_supervised_label_batch,
          train_unsupervised_label_batch) in enumerate(train_dataset):
        train_label_batch = (train_supervised_label_batch,
                             train_unsupervised_label_batch)
        train_batch(train_feature_batch, train_label_batch, encoder, decoder,
                    task, optimizer, train_step, freeze_pretrained_encoder)
        train_step += 1
        # continuous eval.
        if train_step % FLAGS.eval_supervised_every_nsteps == 0:
          _, _, _, metrics = eval_all(eval_dataset, encoder, decoder, task,
                                      'eval', train_step)
          auc, aucpr = metrics
          if auc > best_metrics:
            # Select the best model based on eval dataset AUC and generate final
            # eval result over test dataset.
            best_metrics = auc
            eval_all(test_dataset, encoder, decoder, task, 'test', train_step)


def main(argv):
  del argv
  if FLAGS.fold >= 1:
    mimic_data_path = FLAGS.mimic_data_path + str(FLAGS.fold) + '/'
  else:
    mimic_data_path = FLAGS.mimic_data_path
  train_dataset, eval_dataset, test_dataset, dataset_config = get_datasets(
      FLAGS.dataset_name, mimic_data_path)

  if FLAGS.train_mode == 'supervised':
    train_supervised_model(train_dataset, FLAGS.train_data_percentage,
                           eval_dataset, test_dataset, dataset_config, None,
                           False)
  if FLAGS.train_mode == 'unsupervised':
    train_unsupervised_model(train_dataset, eval_dataset, dataset_config)

  if FLAGS.train_mode == 'pretrain':
    dummy_feature, dummy_supervised_label, dummy_unsupervised_label = next(
        iter(train_dataset.batch(FLAGS.train_batch_size)))

    last_encoder, best_encoder, global_step = train_unsupervised_model(
        train_dataset, eval_dataset, dataset_config)

    if FLAGS.use_best_encoder:
      pretrained_encoder = copy_encoder(best_encoder, dummy_feature)
    else:
      pretrained_encoder = copy_encoder(last_encoder, dummy_feature)

    if FLAGS.transfer_data_path:
      logging.info('end pretraining, start transfer to new task.')
      transfer_train_dataset, transfer_eval_dataset, transfer_test_dataset, transfer_dataset_config = get_datasets(
          FLAGS.dataset_name, FLAGS.transfer_data_path)
      train_supervised_model(transfer_train_dataset,
                             FLAGS.train_data_percentage, transfer_eval_dataset,
                             transfer_test_dataset, transfer_dataset_config,
                             pretrained_encoder,
                             FLAGS.freeze_encoder_after_pretrain, global_step)

    else:
      logging.info('end pretraining, start supervised training.')
      train_supervised_model(train_dataset, FLAGS.train_data_percentage,
                             eval_dataset, test_dataset, dataset_config,
                             pretrained_encoder,
                             FLAGS.freeze_encoder_after_pretrain, global_step)

  if FLAGS.train_mode == 'cotrain_pretrain':
    dummy_feature, _, _ = next(
        iter(train_dataset.batch(FLAGS.train_batch_size)))

    last_encoder, best_encoder, global_step = train_cotrain_unsupervised_model(
        train_dataset, FLAGS.train_data_percentage, eval_dataset, test_dataset,
        dataset_config)

    if FLAGS.use_best_encoder:
      pretrained_encoder = copy_encoder(best_encoder, dummy_feature)
    else:
      pretrained_encoder = copy_encoder(last_encoder, dummy_feature)

    train_supervised_model(train_dataset, FLAGS.train_data_percentage,
                           eval_dataset, test_dataset, dataset_config,
                           pretrained_encoder,
                           FLAGS.freeze_encoder_after_pretrain, global_step)

  if FLAGS.train_mode == 'attn_approx':
    dummy_feature, _, _ = next(
        iter(train_dataset.batch(FLAGS.train_batch_size)))

    last_encoder, best_encoder, global_step = train_supervising_unsupervised_model_attn_approx(
        train_dataset, FLAGS.train_data_percentage, eval_dataset, test_dataset,
        dataset_config)

    if FLAGS.use_best_encoder:
      pretrained_encoder = copy_encoder(best_encoder, dummy_feature)
    else:
      pretrained_encoder = copy_encoder(last_encoder, dummy_feature)

    if FLAGS.transfer_data_path:
      logging.info('end pretraining, start transfer to new task.')
      transfer_train_dataset, transfer_eval_dataset, transfer_test_dataset, transfer_dataset_config = get_datasets(
          FLAGS.dataset_name, FLAGS.transfer_data_path)
      train_supervised_model(transfer_train_dataset,
                             FLAGS.train_data_percentage, transfer_eval_dataset,
                             transfer_test_dataset, transfer_dataset_config,
                             pretrained_encoder,
                             FLAGS.freeze_encoder_after_pretrain, global_step)

    else:
      logging.info('end pretraining, start supervised training.')
      train_supervised_model(train_dataset, FLAGS.train_data_percentage,
                             eval_dataset, test_dataset, dataset_config,
                             pretrained_encoder,
                             FLAGS.freeze_encoder_after_pretrain, global_step)

  if FLAGS.train_mode == 'attn_hessian':
    dummy_feature, _, _ = next(
        iter(train_dataset.batch(FLAGS.train_batch_size)))

    last_encoder, best_encoder, global_step = train_supervising_unsupervised_model_attn_hessian(
        train_dataset, eval_dataset, test_dataset, dataset_config)

    if FLAGS.use_best_encoder:
      pretrained_encoder = copy_encoder(best_encoder, dummy_feature)
    else:
      pretrained_encoder = copy_encoder(last_encoder, dummy_feature)

    train_supervised_model(train_dataset, FLAGS.train_data_percentage,
                           eval_dataset, test_dataset, dataset_config,
                           pretrained_encoder,
                           FLAGS.freeze_encoder_after_pretrain, global_step)


ENCODER_MAP = {
    'lstm': encoder_decoder.LSTMEncoder,
}

DECODER_MAP = {
    'mlp': encoder_decoder.MLPDecoder,
    'lstm': encoder_decoder.LSTMDecoder,
}

if __name__ == '__main__':
  app.run(main)
