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
"""Tasks to be used by unsupervised learning of patient representation."""

import os
import tensorflow as tf

import file
import logging

import customized_layers


def _get_tensorboard_writer(experiment_log_path, job_prefix, mode, task):
  path = os.path.join(experiment_log_path, job_prefix, 'scalars', task, mode)
  return tf.summary.create_file_writer(path)


def output_attention_variable(attention_list, feature_keys, step,
                              experiment_log_path, job_prefix):
  """Ouptput attention var values to CSV file."""
  csvfile = f'{experiment_log_path}/attention_{job_prefix}.csv'
  attention_val_str = ''
  for atten_w in attention_list:
    # Note: there is only one layer in the attention layer definition.
    # Thus it only loops once here.
    attention_val_str = attention_val_str + ','.join([
        str(feature_key) + ':' + str(feature_w)
        for feature_key, feature_w in zip(feature_keys,
                                          atten_w.numpy().tolist())
    ])

  with file.Open(csvfile, 'w+') as csvfile:
    csvfile.write(f'{step}:{attention_val_str}\n')


def output_best_eval_csv(export_metrics, loss, metrics1, metrics2, mode, step,
                         experiment_log_path, job_prefix):
  csvfile = f'{experiment_log_path}/best_{mode}_{export_metrics}_{job_prefix}.csv'
  with file.Open(csvfile, 'wt') as csvfile:
    csvfile.write(f'{loss},{metrics1},{metrics2},{step}\n')


class ClassificationTask(object):
  """A task that handles binary classification loss and metric computation."""

  def __init__(self, log_steps, export_to_tensorboard, experiment_log_path,
               job_prefix):
    super().__init__()
    self.name = 'classification'
    self.log_steps = log_steps

    self.export_to_tensorboard = export_to_tensorboard
    self.experiment_log_path = os.path.join(experiment_log_path)

    self.job_prefix = job_prefix

    self.positive = {'train': 0, 'eval': 0, 'test': 0}
    self.total = {'train': 0, 'eval': 0, 'test': 0}
    self.loss = {'train': 0, 'eval': 0, 'test': 0}
    self.auc = {
        'train': tf.keras.metrics.AUC(),
        'eval': tf.keras.metrics.AUC(),
        'test': tf.keras.metrics.AUC()
    }
    self.aucpr = {
        'train': tf.keras.metrics.AUC(curve='PR'),
        'eval': tf.keras.metrics.AUC(curve='PR'),
        'test': tf.keras.metrics.AUC(curve='PR')
    }

    self.best_eval_loss = 10.0
    self.best_eval_auc = 0
    self.best_eval_aucpr = 0

    self.writer = {
        k: _get_tensorboard_writer(self.experiment_log_path, self.job_prefix, k,
                                   self.name)
        for k in ['train', 'eval', 'test']
    }

  def process_label(self, label):
    supervised_label, _ = label
    supervised_label = tf.expand_dims(supervised_label, axis=1)
    return supervised_label

  def get_loss(self, label, prediction, mode):
    label = self.process_label(label)
    self.loss[mode] = tf.math.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true=label, y_pred=prediction, from_logits=False))

    return self.loss[mode]

  def update_metrics(self, label, prediction, mode, global_step, reset):
    """Update metrics for the task at global_step."""
    label = self.process_label(label)
    if reset:
      self.auc[mode].reset_states()
      self.aucpr[mode].reset_states()
      self.positive[mode] = 0
      self.total[mode] = 0

    self.auc[mode].update_state(y_true=label, y_pred=prediction)
    self.aucpr[mode].update_state(y_true=label, y_pred=prediction)
    self.positive[mode] += tf.math.count_nonzero(label)
    self.total[mode] += label.shape[0]
    pos_ratio = self.positive[mode] / self.total[mode]
    auc = self.auc[mode].result().numpy()
    aucpr = self.aucpr[mode].result().numpy()
    loss = self.loss[mode]

    if self.export_to_tensorboard:
      with self.writer[mode].as_default():
        tf.summary.scalar(mode + '/classification/loss', loss, global_step)
        tf.summary.scalar(mode + '/classification/auc', auc, global_step)
        tf.summary.scalar(mode + '/classification/aucpr', aucpr, global_step)
        tf.summary.scalar(mode + '/classification/positive', pos_ratio,
                          global_step)
        self.writer[mode].flush()

    if global_step % self.log_steps == 0 or mode == 'eval' or mode == 'test':
      logging.info(
          '%s : Step %d: loss = %f, auc = %f aucpr = %f pos_ratio = %f ', mode,
          global_step, loss, auc, aucpr, pos_ratio)

    if mode == 'eval':
      if self.loss[mode] < self.best_eval_loss:
        self.best_eval_loss = loss
        output_best_eval_csv('loss', loss, auc, aucpr, mode, global_step,
                             self.experiment_log_path, self.job_prefix)
      if auc > self.best_eval_auc:
        self.best_eval_auc = auc
        output_best_eval_csv('auc', loss, auc, aucpr, mode, global_step,
                             self.experiment_log_path, self.job_prefix)
      if aucpr > self.best_eval_aucpr:
        self.best_eval_aucpr = aucpr
        output_best_eval_csv('aucpr', loss, auc, aucpr, mode, global_step,
                             self.experiment_log_path, self.job_prefix)

    if mode == 'test':
      output_best_eval_csv('', loss, auc, aucpr, mode, global_step,
                           self.experiment_log_path, self.job_prefix)

    return auc, aucpr


class RegressionTask(object):
  """A task that handles regression loss and metric computation."""

  def __init__(self, num_feature, num_predict, target_feature_index,
               feature_indice_included_str, feature_indice_excluded_str,
               feature_keys, log_steps, export_to_tensorboard,
               experiment_log_path, job_prefix, forecast_only_nonmask_values,
               forecast_loss_func, use_attention, normalization_model,
               attention_init_value, use_attn_output):
    super().__init__()
    self.name = 'regression'
    self.use_attention = use_attention
    if self.use_attention:
      self.attention = customized_layers.AttentionLayer(num_predict,
                                                        normalization_model,
                                                        attention_init_value)
    self.use_attn_output = use_attn_output
    self.num_feature = num_feature
    self.num_predict = num_predict
    self.target_feature_index = target_feature_index
    # Only one list can be nonempty.
    self.feature_keys = feature_keys
    self.feature_index_included = self._parse_feature_index_str(
        feature_indice_included_str, feature_indice_excluded_str)
    if self.feature_index_included:
      logging.info('selected features:')
      for f_index in self.feature_index_included:
        logging.info(feature_keys[f_index])
    else:
      logging.info('all feature included.')

    self.log_steps = log_steps

    self.export_to_tensorboard = export_to_tensorboard
    self.experiment_log_path = os.path.join(experiment_log_path)
    self.job_prefix = job_prefix

    self.forecast_only_nonmask_values = forecast_only_nonmask_values
    self.forecast_loss_func = forecast_loss_func

    self.label_val = {'train': 0, 'eval': 0, 'test': 0}
    self.total = {'train': 0, 'eval': 0, 'test': 0}
    self.loss = {'train': 0, 'eval': 0, 'test': 0}
    self.mse = {
        'train': tf.keras.metrics.MeanSquaredError(),
        'eval': tf.keras.metrics.MeanSquaredError(),
        'test': tf.keras.metrics.MeanSquaredError()
    }
    self.mae = {
        'train': tf.keras.metrics.MeanAbsoluteError(),
        'eval': tf.keras.metrics.MeanAbsoluteError(),
        'test': tf.keras.metrics.MeanAbsoluteError()
    }

    self.best_eval_loss = 10.0
    self.best_eval_mse = 10.0
    self.best_eval_mae = 10.0

    if self.use_attention:
      for f in self.feature_keys:
        key = 'attention/' + f

    self.writer = {
        k: _get_tensorboard_writer(self.experiment_log_path, self.job_prefix, k,
                                   self.name)
        for k in ['train', 'eval', 'test']
    }

  def process_label(self, label):
    _, unsupervised_label = label
    return unsupervised_label

  def get_loss(self, label, prediction, mode):
    """Loss for regression task."""
    future_obs, future_mask = self.process_label(label)

    if self.target_feature_index < self.num_feature:
      # target_feature_index [0, num_feature) for single feature forecast task.
      future_obs = future_obs[:, :, self.target_feature_index]
      prediction = prediction[:, :, self.target_feature_index]
      future_mask = future_mask[:, :, self.target_feature_index]

    if self.feature_index_included:
      future_obs = tf.gather(future_obs, self.feature_index_included, axis=2)
      prediction = tf.gather(prediction, self.feature_index_included, axis=2)
      future_mask = tf.gather(future_mask, self.feature_index_included, axis=2)

    loss_func = tf.abs if self.forecast_loss_func == 'abs' else tf.square
    if not self.forecast_only_nonmask_values:
      future_mask = tf.ones_like(future_mask)

    if self.use_attention:
      if self.feature_index_included:
        logging.fatal('use_attention can not work with feature_index_included')
      future_obs = self.attention(future_obs)
      prediction = self.attention(prediction)

    # shape is (?,), where ? is within [0, num_predict * num_feature]
    # depending on the number of masks.
    batch_loss_with_mask = tf.boolean_mask(
        loss_func(future_obs - prediction), future_mask)
    self.loss[mode] = tf.reduce_mean(batch_loss_with_mask)

    return self.loss[mode]

  def update_metrics(self, label, prediction, mode, global_step, reset):
    """Update metrics for the task at global_step."""
    future_obs, future_mask = self.process_label(label)

    if self.target_feature_index < self.num_feature:
      # target_feature_index [0, num_feature) for single feature forecast task.
      future_obs = future_obs[:, :, self.target_feature_index]
      prediction = prediction[:, :, self.target_feature_index]
      future_mask = future_mask[:, :, self.target_feature_index]

    if not self.forecast_only_nonmask_values:
      future_mask = tf.ones_like(future_mask)

    future_obs = tf.boolean_mask(future_obs, future_mask)
    prediction = tf.boolean_mask(prediction, future_mask)

    if reset:
      self.mse[mode].reset_states()
      self.mae[mode].reset_states()
      self.label_val[mode] = 0
      self.total[mode] = 0

    self.mse[mode].update_state(y_true=future_obs, y_pred=prediction)
    self.mae[mode].update_state(y_true=future_obs, y_pred=prediction)
    self.total[mode] += future_obs.shape[0]
    self.label_val[mode] += tf.reduce_sum(future_obs)
    avg_feature_label_value = self.label_val[mode] / self.total[
        mode] / self.num_predict / self.num_feature
    mse = self.mse[mode].result().numpy()
    mae = self.mae[mode].result().numpy()
    loss = self.loss[mode]

    if self.export_to_tensorboard:
      with self.writer[mode].as_default():
        tf.summary.scalar(mode + '/regression/loss', loss, global_step)
        tf.summary.scalar(mode + '/regression/mse', mse, global_step)
        tf.summary.scalar(mode + '/regression/mae', mae, global_step)
        tf.summary.scalar(mode + '/regression/label_val',
                          avg_feature_label_value, global_step)
        if self.use_attention:
          attn_list = self.attention_outputs if self.use_attn_output else self.attention_variables
          for atten_w in attn_list:
            for j, feature_w in enumerate(atten_w.numpy().tolist()):
              tf.summary.scalar('attention/' + self.feature_keys[j], feature_w,
                                global_step)
      self.writer[mode].flush()

    if global_step % self.log_steps == 0 or mode == 'eval' or mode == 'test':
      logging.info(
          '%s : Step %d: loss = %f, mse = %f, mae = %f, label_val = %f ', mode,
          global_step, loss, mse, mae, avg_feature_label_value)

    if mode == 'eval':
      if self.use_attention:
        attn_list = self.attention_outputs if self.use_attn_output else self.attention_variables
        for atten_w in attn_list:
          for j, feature_w in enumerate(atten_w.numpy().tolist()):
            key = 'attention/' + self.feature_keys[j]

        for atten_w in attn_list:
          attention_log_str = [
              'attn/' + self.feature_keys[j] + ':' + str(feature_w)
              for j, feature_w in enumerate(atten_w.numpy().tolist())
          ]
          logging.info(attention_log_str)
        output_attention_variable(attn_list, self.feature_keys, global_step,
                                  self.experiment_log_path, self.job_prefix)

    if mode == 'eval':
      if loss < self.best_eval_loss:
        self.best_eval_loss = loss
        output_best_eval_csv('loss', loss, mse, mae, mode, global_step,
                             self.experiment_log_path, self.job_prefix)
      if mse < self.best_eval_mse:
        self.best_eval_mse = mse
        output_best_eval_csv('mse', loss, mse, mae, mode, global_step,
                             self.experiment_log_path, self.job_prefix)
      if mae < self.best_eval_mae:
        self.best_eval_mae = mae
        output_best_eval_csv('mae', loss, mse, mae, mode, global_step,
                             self.experiment_log_path, self.job_prefix)

    if mode == 'test':
      output_best_eval_csv('', loss, mse, mae, mode, global_step,
                           self.experiment_log_path, self.job_prefix)

    return mse, mae

  def _parse_feature_index_str(self, feature_indice_included_str,
                               feature_indice_excluded_str):
    """Parse feature index string to list."""
    feature_index_included = []
    if feature_indice_included_str and feature_indice_excluded_str:
      raise ValueError(
          'Only one of feature_indice_included_str and feature_indice_excluded_str can be non-empty.'
      )
    if not feature_indice_included_str and not feature_indice_excluded_str:
      logging.info('No manual feature selection. All feature included.')
      return feature_index_included

    if feature_indice_included_str:
      feature_index_included = feature_indice_included_str.split(',')
      if not all(
          int(i) >= 0 and int(i) < self.num_feature
          for i in feature_index_included):
        raise ValueError(
            f'values in feature_indice_included_str needs to be within [0, {self.num_feature}).'
        )
      feature_index_included = [int(i) for i in feature_index_included]
    if feature_indice_excluded_str:
      feature_index_excluded = feature_indice_excluded_str.split(',')
      if not all(
          int(i) >= 0 and int(i) < self.num_feature
          for i in feature_index_excluded):
        raise ValueError(
            f'values in  feature_indice_excluded_str needs to be within [0, {self.num_feature}).'
        )
      feature_index_included = [
          f for f in range(self.num_feature)
          if str(f) not in feature_index_excluded
      ]

    return feature_index_included

  @property
  def attention_variables(self):
    if self.use_attention:
      return self.attention.trainable_variables
    else:
      raise ValueError(
          'use pretrain_attention mode to access attention_variables')

  @property
  def attention_outputs(self):
    if self.use_attention:
      return self.attention.attention_outputs
    else:
      raise ValueError('use attn_* mode to access attention_output')
