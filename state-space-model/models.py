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
"""A library of models."""
from typing import Dict

import experiment_config_pb2
import modules
import multi_head_for_survival as multi_head
import sequence_heads
import survival_heads
import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow_probability as tfp

MultivariateNormalTriL = tfp.distributions.MultivariateNormalTriL
PARALLEL_ITER_SCAN = 10


class Model(object):
  """Base model class."""

  def __init__(self,
               mode: tf.Estimator.ModeKeys,
               config: experiment_config_pb2.ModelConfig = None):
    """Initialize model and parse config.

    Args:
      mode: One of {TRAIN,EVAL}.
      config: Model config.
    """
    self._mode = mode
    self._config = self._parse_config(config)

  def _parse_config(self, config: experiment_config_pb2.ModelConfig = None):
    """Update default config based on the supplied model configuration."""
    default_config = experiment_config_pb2.ModelConfig()
    if config:
      default_config.MergeFrom(config)
    return default_config

  @property
  def config(self):
    return self._config

  def _feature_to_tensor(self, features: Dict[bytes, tf.Tensor]):
    """Convert input features into dense tensors.

    Args:
      features: Mapping of feature names to tensors. Each training tensor has
        shape [batch_size, context_len_to_trigger]. Each pretrain tensor has
        shape [batch_size, context_window_size].

    Returns:
      A dict of the following tensors:
        obs_full_tensor of shape [batch_size, context_window_size, num_obs]
        obs_full_mask_tensor of shape [batch_size, context_window_size, num_obs]
        obs_to_trigger_tensor of shape [bs, context_len_to_trigger, num_obs]
        obs_to_trigger_mask_tensor shape [bs, context_len_to_trigger, num_obs]
        intervention_full_tensor of shape [bs, context_window_size, num_int]
        intervention_to_trigger_tensor : [bs, context_len_to_trigger, num_int]
    """
    config = self._config
    tensor_dict = {}
    tf.logging.info(features.keys())
    tensor_dict['obs_full_tensor'] = tf.concat(
        [tf.expand_dims(features[obs], 2) for obs in config.observation_codes],
        axis=2)
    if config.has_mask:
      mask_list = []
      for obs in config.observation_codes:
        if obs.endswith('_raw'):
          obs = obs[:-4]
        mask_list = mask_list + [tf.expand_dims(features[obs + '_mask'], 2)]
      tensor_dict['obs_full_mask_tensor'] = tf.concat(mask_list, axis=2)

    tensor_dict['obs_to_trigger_tensor'] = tf.slice(
        tensor_dict['obs_full_tensor'], [0, 0, 0],
        [-1, self._config.context_len_to_trigger, -1])

    if config.has_mask:
      tensor_dict['obs_to_trigger_mask_tensor'] = tf.slice(
          tensor_dict['obs_full_mask_tensor'], [0, 0, 0],
          [-1, self._config.context_len_to_trigger, -1])

    if config.intervention_codes:
      tensor_dict['intervention_full_tensor'] = tf.concat([
          tf.expand_dims(features[intv], 2)
          for intv in config.intervention_codes
      ],
                                                          axis=2)
      tensor_dict['intervention_to_trigger_tensor'] = tf.slice(
          tensor_dict['intervention_full_tensor'], [0, 0, 0],
          [-1, self._config.context_len_to_trigger, -1])
    else:
      tensor_dict['intervention_full_tensor'] = None
      tensor_dict['intervention_to_trigger_tensor'] = None

    if config.forecast_biomarkers:
      biomarker_mask_list = []
      for obs in config.observation_codes:
        if obs in config.forecast_biomarkers:
          biomarker_mask_list.append(True)
        else:
          biomarker_mask_list.append(False)
      tensor_dict['biomarker_boolean_mask'] = tf.stack(
          biomarker_mask_list, axis=0)
    else:
      tensor_dict['biomarker_boolean_mask'] = tf.constant(
          True, shape=[len(config.observation_codes)])

    if 'true_length_hr' in features:
      tensor_dict['true_length_hr'] = features['true_length_hr']
    return tensor_dict

  def _get_train_and_predict_tensors(self, tensor_dict, train_len):
    obs_full_tensor = tensor_dict['obs_full_tensor']
    intervention_full_tensor = tensor_dict['intervention_full_tensor']
    if self._config.has_mask:
      obs_full_mask_tensor = tf.cast(tensor_dict['obs_full_mask_tensor'],
                                     tf.float32)
    else:
      obs_full_mask_tensor = tf.ones_like(obs_full_tensor, dtype=tf.float32)

    if train_len > 0:
      obs_full_tensor = tf.slice(obs_full_tensor, [0, 0, 0],
                                 [-1, train_len, -1])
      intervention_full_tensor = tf.slice(intervention_full_tensor, [0, 0, 0],
                                          [-1, train_len, -1])
      obs_full_mask_tensor = tf.slice(obs_full_mask_tensor, [0, 0, 0],
                                      [-1, train_len, -1])

    obs_to_trigger_tensor = tensor_dict['obs_to_trigger_tensor']
    intervention_to_trigger_tensor = tensor_dict[
        'intervention_to_trigger_tensor']
    if self._config.has_mask:
      obs_to_trigger_mask_tensor = tf.cast(
          tensor_dict['obs_to_trigger_mask_tensor'], tf.float32)
    else:
      obs_to_trigger_mask_tensor = tf.ones_like(
          obs_to_trigger_tensor, dtype=tf.float32)

    if self._config.forecast_biomarkers:
      biomarker_boolean_mask_tensor = tensor_dict['biomarker_boolean_mask']
    else:
      biomarker_boolean_mask_tensor = tf.constant(
          True, shape=[len(self._config.observation_codes)])

    return (obs_full_tensor, obs_full_mask_tensor, intervention_full_tensor,
            obs_to_trigger_tensor, obs_to_trigger_mask_tensor,
            intervention_to_trigger_tensor, biomarker_boolean_mask_tensor)

  def _encode(self, tensor_dict: Dict[bytes, tf.Tensor]) -> tf.Tensor:
    """Encode input features into a fixed length representation.

    This method will be implemented in subclasses based different models.

    Args:
      tensor_dict: dict of the following tensors:
      * obs_full_tensor of shape [batch_size, context_window_size, num_obs]
        representing the input observation features. This is the feature for
        pretraining.
      * obs_full_mask_tensor of shape [batch_size, context_window_size, num_obs]
      * obs_to_trigger_tensor of shape [bs, context_len_to_trigger, num_obs]
        representing the input observation features up to trigger time. This is
        the feature used for training.
      * obs_to_trigger_mask_tensor shape [bs, context_len_to_trigger, num_obs]
      * intervention_full_tensor of shape [bs, context_window_size, num_int]
        representing the input interv features. This is the feature for
        pretraining.
      * intervention_to_trigger_tensor : [bs, context_len_to_trigger, num_int]
        representing the input interv features up to trigger time. This is
        the feature used for training.
    Returns:
      A tensor of shape [batch_size, output_dimension] where output_dimension is
      the output dimension depending on the model config.
    """
    raise NotImplementedError()

  def generate_output(self, features: Dict[bytes, tf.Tensor]):
    """Generate model output."""
    with tf.name_scope('Feature_to_tensor'):
      tensor_dict = self._feature_to_tensor(features)

    tf.logging.info(tensor_dict)
    with tf.name_scope(self.__class__.__name__):
      return self._encode(tensor_dict)


class MultiLayerPerceptron(Model):
  """Multi-layer perceptron (MLP) model."""

  def __init__(self, mode, config=None):
    """Initialize an MLP model."""
    super(MultiLayerPerceptron, self).__init__(mode, config=config)

  def _encode(self, tensor_dict: Dict[bytes, tf.Tensor]) -> tf.Tensor:
    """Encode the most recent inputs of the input tensor based on MLP model.

    Args:
      tensor_dict: A dict of tensor.
    Returns:
      dict of network output tensor with shape [batch_size, self._config.sdl].
    """
    obs_tensor = tensor_dict['obs_to_trigger_tensor']
    config = self._config
    is_training = self._mode == tf.estimator.ModeKeys.TRAIN
    network_input = modules._recent_input_module(obs_tensor)

    output = modules._fully_connected_module(
        input_tensor=network_input,
        num_dense_layers=config.ndl,
        dense_layer_size=config.sdl,
        output_layer_size=config.sdl,
        output_activation=tf.nn.sigmoid,
        is_training=is_training,
        drop_rate=config.drdl,
        name='MLP',
    )
    output_dict = {}
    output_dict['state_encoding'] = output
    return output_dict


class MostRecentInput(Model):
  """Most recent input (MRI) model."""

  def __init__(self, mode, config=None):
    """Initialize a MRI model."""
    super(MostRecentInput, self).__init__(mode, config=config)

  def _encode(self, tensor_dict: Dict[bytes, tf.Tensor]) -> tf.Tensor:
    """Encode the most recent inputs of the input tensor.

    Args:
      tensor_dict: A dict of tensor.

    Returns:
      dict of network output tensor with shape [batch_size, self._config.sdl].
    """
    tf.logging.info(tensor_dict)
    obs_tensor = tensor_dict['obs_to_trigger_tensor']

    output = {}
    output['state_encoding'] = modules._recent_input_module(obs_tensor)
    return output


class LSTMDynamicSystem(Model):
  """Dynamic system model based on LSTM."""

  def __init__(self, mode, config=None):
    """Initialize a LSTM DS model."""
    super(LSTMDynamicSystem, self).__init__(mode, config=config)
    if self._config.forecast_biomarkers:
      self._out_obs_dim = len(self._config.forecast_biomarkers)
    else:
      self._out_obs_dim = len(self._config.observation_codes)
    if self._config.reuse_encoding:
      self._tag = ''
    else:
      self._tag = 'encode'

  def lstm_output_state(self, input_tensor, seqlen):
    """Get output and state from LSTM."""
    config = self._config
    batch_size = input_tensor.get_shape().as_list()[0]

    with tf.variable_scope('lstm_ds' + self._tag, reuse=tf.AUTO_REUSE):
      cell = tf.nn.rnn_cell.LSTMCell(config.ds_state)
      rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * config.ds_nrl)
      initial_state = rnn_cells.zero_state(batch_size, dtype=tf.float32)
      outputs, state = tf.nn.dynamic_rnn(
          rnn_cells,
          input_tensor,
          initial_state=initial_state,
          sequence_length=seqlen,
          dtype=tf.float32)
    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    # 'state' is a tensor of shape [batch_size, tuple<hidden, cell_state_size>]
    return outputs, state

  def lstm_obs_emit_step(self, previous_output, current_input):
    del previous_output
    lstm_output = current_input
    with tf.variable_scope('lstm_ds_obs_emit' + self._tag, reuse=tf.AUTO_REUSE):
      obs_estimation = tf.layers.dense(
          inputs=lstm_output,
          units=self._out_obs_dim,
          activation=None,
          name='obs_dense')
    return obs_estimation

  def lstm_emit_obs(self, seq_output):
    """Emit observations based on seq output from lstm."""
    # seq_output shape [bs, tlen, lstm_state]
    batch_size = seq_output.get_shape().as_list()[0]
    time_first_input = tf.transpose(seq_output, [1, 0, 2])
    init_obs = tf.zeros([batch_size, self._out_obs_dim])
    current_obs = tf.scan(
        self.lstm_obs_emit_step,
        time_first_input,
        initializer=(init_obs),
        parallel_iterations=PARALLEL_ITER_SCAN,
        name='lstm_emit_obs_scan')
    # switch batch, tlen dimension back.
    return tf.transpose(current_obs, [1, 0, 2])

  def _encode(self, tensor_dict: Dict[bytes, tf.Tensor]) -> tf.Tensor:
    """Encode an input tensor based on LSTM model.

    Args:
      tensor_dict: A dict of tensor.

    Returns:
      dict of network output tensor.
    """
    config = self._config

    train_and_predict_tensors = self._get_train_and_predict_tensors(
        tensor_dict, config.sys_id_len)
    _, _, _, obs_to_trigger_tensor, obs_to_trigger_mask_tensor, intervention_to_trigger_tensor, biomarker_boolean_mask_tensor = train_and_predict_tensors  # pylint:

    batch_size = obs_to_trigger_tensor.get_shape().as_list()[0]
    context_size = obs_to_trigger_tensor.get_shape().as_list()[1]
    num_obs = obs_to_trigger_tensor.get_shape().as_list()[2]

    seqlen = tf.constant(context_size, shape=[batch_size])

    tf.logging.info(obs_to_trigger_tensor)
    tf.logging.info(intervention_to_trigger_tensor)
    if intervention_to_trigger_tensor is not None:
      if config.lstm_interv_delay == 0:
        input_tensor = tf.concat([
            obs_to_trigger_tensor,
            intervention_to_trigger_tensor],
                                 axis=2)
      else:
        input_tensor = tf.concat([
            obs_to_trigger_tensor[:, config.lstm_interv_delay:, :],
            intervention_to_trigger_tensor[:, :-config.lstm_interv_delay, :]
        ],
                                 axis=2)
    else:
      input_tensor = obs_to_trigger_tensor

    # 'seq_output' is a tensor of shape [batch_size, seq_len, cell_state_size]
    # 'state' is a tuple of
    # (LSTMStateTuple(c=<tf.Tensor shape=(batch_size, cell_state_size)>,
    #                 h=<tf.Tensor shape=(batch_size, cell_state_size)>),)
    seq_output, state = self.lstm_output_state(input_tensor, seqlen)
    obs_estimation = self.lstm_emit_obs(seq_output)

    output = dict()
    # obs estimation time series shape [bs, context_size, num_obs]
    output['obs_est'] = tf.clip_by_value(
        obs_estimation, -99999, 99999, name=None)

    # final state_encoding shape [bs, ds_state]
    state_encoding = seq_output[:, -1, :]
    output['state_encoding'] = tf.clip_by_value(
        state_encoding, -99999, 99999, name=None)
    # output['lstm_forecast_state'] shape [bs, 2, cell_state_size]
    output['lstm_forecast_state'] = tf.stack([state[0].c, state[0].h], axis=1)
    tf.logging.info(output['lstm_forecast_state'])

    # output['last_obs']  shape [bs, _a_dim]
    output['last_obs'] = tf.squeeze(
        tf.slice(obs_to_trigger_tensor,
                 [0, self._config.context_len_to_trigger - 1, 0], [-1, 1, -1]))
    if self._config.forecast_biomarkers:
      # switch shape to [_a_dim, bs] for applying mask.
      output['last_obs'] = tf.boolean_mask(
          tf.transpose(output['last_obs']), biomarker_boolean_mask_tensor)
      # transpose shape back.
      output['last_obs'] = tf.transpose(output['last_obs'])

    return output


class DKFDynamicSystem(Model):
  """dynamic system model based on Deep Kalman Filter.

    Deep Kalman Filters: https://arxiv.org/abs/1511.05121
    Structured Inference Networks for Nonlinear State Space Models:
      https://arxiv.org/abs/1609.09869
  """

  def __init__(self, mode, config=None):
    super(DKFDynamicSystem, self).__init__(mode, config=config)
    self._z_dim = config.ds_state
    if self._config.reuse_encoding:
      self._tag = ''
    else:
      self._tag = 'encode'

    self._a_dim = len(config.observation_codes)
    if self._config.forecast_biomarkers:
      self._out_obs_dim = len(self._config.forecast_biomarkers)
    else:
      self._out_obs_dim = len(self._config.observation_codes)

    if not config.intervention_codes:
      raise ValueError('missing intervention_codes')
    else:
      self._u_dim = len(config.intervention_codes)
    # state noise.
    self.state_noise = tf.get_variable(
        'state_noise',
        initializer=tf.eye(self._z_dim, num_columns=self._z_dim),
        trainable=config.noise_trainable)

    # observation noise.
    self.obs_noise = tf.get_variable(
        'obs_noise',
        initializer=tf.eye(self._out_obs_dim, num_columns=self._out_obs_dim),
        trainable=config.noise_trainable)  # state uncertainty

    # intervention forecast noise.
    self.interv_noise = tf.get_variable(
        'interv_noise',
        initializer=tf.eye(self._u_dim, num_columns=self._u_dim),
        trainable=config.noise_trainable)

  def inference_network(self, obs_tensor, intervention_tensor, mask_tensor,
                        infer_model):
    if infer_model == 'basic':
      return self.inference_network_basic(obs_tensor, intervention_tensor,
                                          mask_tensor)
    else:
      return self.inference_network_struct(obs_tensor, intervention_tensor,
                                           mask_tensor, infer_model)

  def get_mu_sigma_sq(self, outputs, batch_size, context_size, name):
    # Pass RNN output to a linear layer then reshape to match.
    # mu_smooth shape [bs, tlen, _z_dim]
    outputs = tf.reshape(outputs, (batch_size * context_size, -1))
    with tf.variable_scope('get_mu_sigma_sq' + name, reuse=tf.AUTO_REUSE):
      mu_smooth = tf.layers.dense(
          inputs=outputs,
          units=self._z_dim,
          activation=None,
          name='mu_smooth' + name)
      mu_smooth = tf.reshape(mu_smooth, [batch_size, context_size, self._z_dim])

      # shape [tlen, bs,  _z_dim, _z_dim]
      sigma_smooth = tf.math.softplus(
          tf.layers.dense(
              inputs=outputs,
              units=self._z_dim * self._z_dim,
              activation=None,
              name='sigma_smooth' + name))
      sigma_smooth = tf.reshape(
          sigma_smooth, [batch_size, context_size, self._z_dim, self._z_dim])

    states = (mu_smooth, sigma_smooth)
    return states

  def get_mu_sigma(self, outputs, batch_size, context_size, name):
    # Pass RNN output to a linear layer then reshape to match.
    # mu_smooth shape [bs, tlen, _z_dim]
    outputs = tf.reshape(outputs, (batch_size * context_size, -1))
    with tf.variable_scope('get_mu_sigma' + name, reuse=tf.AUTO_REUSE):
      mu_smooth = tf.layers.dense(
          inputs=outputs,
          units=self._z_dim,
          activation=None,
          name='mu_smooth' + name)
      mu_smooth = tf.reshape(mu_smooth, [batch_size, context_size, self._z_dim])

      # shape [tlen, bs,  _z_dim]
      sigma_smooth = tf.math.softplus(
          tf.layers.dense(
              inputs=outputs,
              units=self._z_dim,
              activation=None,
              name='sigma_smooth' + name))
    sigma_smooth = tf.reshape(sigma_smooth,
                              [batch_size, context_size, self._z_dim])

    states = (mu_smooth, sigma_smooth)
    return states

  def inference_network_struct(self, obs_tensor, intervention_tensor,
                               mask_tensor, infer_model):
    """Inference/Recognition network for encoding obs/interv to states.

    Args:
      obs_tensor: tensor with shape [bs, tlen, _a_dim].
      intervention_tensor: tensor with shape [bs, tlen, _u_dim].
      mask_tensor: tensor with shape [bs, tlen, _a_dim].
      infer_model:

    Returns:
      states: a tuple with mu_smooth, sigma_smooth.
      states[0] is mu_smooth with shape [bs, tlen, _z_dim]
      states[1] is sigma_smooth with shape [bs, tlen, _z_dim, _z_dim]
    """
    if 'l' not in infer_model and 'r' not in infer_model:
      raise ValueError('model name missing direction %s' % infer_model)

    del mask_tensor
    config = self._config
    batch_size = obs_tensor.get_shape().as_list()[0]
    context_size = obs_tensor.get_shape().as_list()[1]

    if intervention_tensor is not None:
      input_tensor = tf.concat([obs_tensor, intervention_tensor], axis=2)
    else:
      input_tensor = obs_tensor
    seqlen = tf.constant(context_size, shape=[batch_size])

    if 'l' in infer_model:
      # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
      # 'state' is a tensor of shape [batch_size, cell_state_size]
      with tf.variable_scope('dkf_ds_strl_forward', reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.LSTMCell(config.ds_state)
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * config.ds_nrl)
        initial_state = rnn_cells.zero_state(batch_size, dtype=tf.float32)
        forward_outputs, _ = tf.nn.dynamic_rnn(
            rnn_cells,
            input_tensor,
            initial_state=initial_state,
            sequence_length=seqlen,
            dtype=tf.float32)

    if 'r' in infer_model:
      with tf.variable_scope('dkf_ds_strl_backward', reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.LSTMCell(config.ds_state)
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * config.ds_nrl)
        initial_state = rnn_cells.zero_state(batch_size, dtype=tf.float32)
        backward_outputs, _ = tf.nn.dynamic_rnn(
            rnn_cells,
            tf.reverse(input_tensor, axis=[1]),
            initial_state=initial_state,
            sequence_length=seqlen,
            dtype=tf.float32)

    if 'st' in infer_model:
      def combine_step_fn(params, inputs):
        # z_tm1 shape [bs, self._z_dim]
        t, z_tm1 = params
        forward_outputs = tf.slice(inputs, [0, 0, 0], [-1, -1, self._z_dim])
        backward_outputs = tf.slice(inputs, [0, 0, self._z_dim],
                                    [-1, -1, self._z_dim])

        h_comb = tf.zeros([batch_size, self._z_dim])
        count = 1
        if 'l' in infer_model:
          h_comb = h_comb + tf.squeeze(
              tf.slice(forward_outputs, [0, 0, 0], [-1, t, -1]), axis=1)
          count = count + 1
        if 'r' in infer_model:
          h_comb = h_comb + tf.squeeze(
              tf.slice(backward_outputs, [0, 0, 0], [-1, t, -1]), axis=1)
          count = count + 1
        z_transform = modules._fully_connected_module(
            input_tensor=z_tm1,
            num_dense_layers=1,
            dense_layer_size=self._config.obs_smlp,
            output_layer_size=self._z_dim,
            output_activation=tf.nn.sigmoid,
            is_training=True,
            drop_rate=0,
            name='z_transform')

        h_comb = tf.multiply(1.0 / count, z_transform + h_comb)
        mu_combine, sigma_combine = self.get_mu_sigma_sq(
            h_comb, batch_size, 1, 'st_scan')
        mu_combine = tf.squeeze(mu_combine, axis=1)
        sigma_combine = tf.squeeze(sigma_combine, axis=1)

        #jitter = 1e-2 * tf.eye(
        #    tf.shape(sigma_combine)[-1],
        #    batch_shape=tf.shape(sigma_combine)[0:-2])
        mvn_combine = tfp.distributions.MultivariateNormalTriL(
            mu_combine, sigma_combine)  # + jitter)
        z_t = mvn_combine.sample()
        return t, z_t

      time_first_forward_outputs = tf.transpose(forward_outputs, [1, 0, 2])
      time_first_backward_outputs = tf.transpose(backward_outputs, [1, 0, 2])
      z_0 = tf.zeros([batch_size, self._z_dim])
      #combined_output = tf.scan(
      #    combine_step_fn,
      #    (time_first_forward_outputs, time_first_backward_outputs),
      #    initializer=(0, z_0),
      #    parallel_iterations=PARALLEL_ITER_SCAN,
      #    name='combine')
      combined_output = forward_outputs + backward_outputs
      # time, state = combined_output
      state = tf.transpose(combined_output, [1, 0, 2])
      combined_states = self.get_mu_sigma_sq(state, batch_size, context_size,
                                             'st')
      tf.logging.info(combined_states)
      return combined_states
    else:
      if 'l' in infer_model:
        l_state = self.get_mu_sigma(forward_outputs, batch_size, context_size,
                                    'l')
        if 'r' not in infer_model:
          return l_state
      if 'r' in infer_model:
        r_state = self.get_mu_sigma(backward_outputs, batch_size, context_size,
                                    'r')
        if 'l' not in infer_model:
          return r_state
      mu_combine = tf.divide(
          tf.multiply(l_state[0], r_state[1]) +
          tf.multiply(r_state[0], l_state[1]), l_state[1] + r_state[1])
      sigma_combine = tf.divide(
          tf.multiply(l_state[1], r_state[1]), l_state[1] + r_state[1])
      sigma_combine_exp = tf.reshape(
          tf.tile(sigma_combine, [1, 1, self._z_dim]),
          [batch_size, context_size, self._z_dim, self._z_dim])
      tf.logging.info(mu_combine, sigma_combine_exp)
      return (mu_combine, sigma_combine_exp)

  def inference_network_basic(self, obs_tensor, intervention_tensor,
                              mask_tensor):
    """Inference/Recognition network for encoding obs/interv to states.

    Args:
      obs_tensor: tensor with shape [bs, tlen, _a_dim].
      intervention_tensor: tensor with shape [bs, tlen, _u_dim].
      mask_tensor: tensor with shape [bs, tlen, _a_dim].

    Returns:
      states: a tuple with mu_smooth, sigma_smooth.
      states[0] is mu_smooth with shape [bs, tlen, _z_dim]
      states[1] is sigma_smooth with shape [bs, tlen, _z_dim, _z_dim]
    """
    del mask_tensor
    config = self._config
    batch_size = obs_tensor.get_shape().as_list()[0]
    context_size = obs_tensor.get_shape().as_list()[1]

    if intervention_tensor is not None:
      input_tensor = tf.concat([obs_tensor, intervention_tensor], axis=2)
    else:
      input_tensor = obs_tensor
    seqlen = tf.constant(context_size, shape=[batch_size])

    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    # 'state' is a tensor of shape [batch_size, cell_state_size]
    with tf.variable_scope('dkf_ds_encode', reuse=tf.AUTO_REUSE):
      cell = tf.nn.rnn_cell.LSTMCell(config.ds_state)
      rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * config.ds_nrl)
      initial_state = rnn_cells.zero_state(batch_size, dtype=tf.float32)
      outputs, _ = tf.nn.dynamic_rnn(
          rnn_cells,
          input_tensor,
          initial_state=initial_state,
          sequence_length=seqlen,
          dtype=tf.float32)

    return self.get_mu_sigma_sq(outputs, batch_size, context_size, 'basic')

  def interv_forecast(self, state):
    """Intervention forecast function.

    Args:
      state: state sample with shape [bs, tlen, self._z_dim].

    Returns:
      interv: interv mean at shape [bs, tlen, self._u_dim].
    """

    def interv_forecast_step_fn(previous_output, current_input):
      del previous_output
      current_state = current_input
      tf.logging.info(current_state)
      next_interv = modules._interv_forecast_module(current_state, self._u_dim,
                                                    self._config.interv_nmlp,
                                                    self._config.interv_smlp,
                                                    self._tag)
      return next_interv

    batch_size = state.get_shape().as_list()[0]
    init_interv = tf.zeros([batch_size, self._u_dim])
    # switch batch, tlen dimension for tf.scan.
    time_first_state = tf.transpose(state, [1, 0, 2])

    next_interv = tf.scan(
        interv_forecast_step_fn,
        time_first_state,
        initializer=(init_interv),
        parallel_iterations=PARALLEL_ITER_SCAN,
        name='interv_forecast_scan')
    # switch batch, tlen dimension back.
    return tf.transpose(next_interv, [1, 0, 2])

  def obs_emission(self, state):
    """Observation emission function.

    Args:
      state: state sample with shape [bs, tlen, self._z_dim].

    Returns:
      obs: obs mean at shape [bs, tlen, self._a_dim].
    """

    def obs_emission_step_fn(previous_output, current_input):
      del previous_output
      current_state = current_input
      tf.logging.info(current_state)
      if self._config.em_act == 'sigmoid':
        emission_activation = tf.nn.sigmoid
      else:
        emission_activation = None

      current_obs = modules._obs_emission_module(current_state,
                                                 self._out_obs_dim,
                                                 self._config.obs_nmlp,
                                                 self._config.obs_smlp,
                                                 self._tag, emission_activation)
      return current_obs

    batch_size = state.get_shape().as_list()[0]
    init_obs = tf.zeros([batch_size, self._out_obs_dim])
    # switch batch, tlen dimension for tf.scan.
    time_first_state = tf.transpose(state, [1, 0, 2])

    current_obs = tf.scan(
        obs_emission_step_fn,
        time_first_state,
        initializer=(init_obs),
        parallel_iterations=PARALLEL_ITER_SCAN,
        name='obs_emission_scan')
    # switch batch, tlen dimension back.
    return tf.transpose(current_obs, [1, 0, 2])

  def state_tran(self, state):
    """State transition function.

    Args:
      state: state sample at t-1 shape [bs, (tlen-1), self._z_dim].

    Returns:
      state: state mean at t  shape [bs, (tlen-1), self._z_dim].
    """

    def state_tran_step_fn(previous_output, current_input):
      del previous_output
      current_state = current_input
      next_state = modules._state_tran_module(current_state, self._z_dim,
                                              self._config.stran_nmlp,
                                              self._config.stran_smlp,
                                              self._tag)
      return next_state

    batch_size = state.get_shape().as_list()[0]
    init_mu = tf.zeros([batch_size, self._z_dim])
    # switch batch, tlen dimension for tf.scan.
    time_first_state = tf.transpose(state, [1, 0, 2])

    next_state = tf.scan(
        state_tran_step_fn,
        time_first_state,
        initializer=(init_mu),
        parallel_iterations=PARALLEL_ITER_SCAN,
        name='state_tran_scan')
    # switch batch, tlen dimension back.
    return tf.transpose(next_state, [1, 0, 2])

  def control_tran(self, interv):
    """Control transition function.

    Args:
      interv: interv at t shape [bs, (tlen-1), self._u_dim].

    Returns:
      state: flattened state mean at t  shape [bs, (tlen-1), self._z_dim].
    """

    def control_tran_step_fn(previous_output, current_input):
      del previous_output
      current_interv = current_input
      # pylint: disable=protected-access
      next_state = modules._control_tran_module(
          current_interv, self._z_dim,
          self._config.ctran_nmlp,
          self._config.ctran_smlp,
          self._tag)
      return next_state

    batch_size = interv.get_shape().as_list()[0]
    init_mu = tf.zeros([batch_size, self._z_dim])
    # switch batch, tlen dimension for tf.scan.
    time_first_interv = tf.transpose(interv, [1, 0, 2])

    next_state = tf.scan(
        control_tran_step_fn,
        time_first_interv,
        initializer=(init_mu),
        parallel_iterations=PARALLEL_ITER_SCAN,
        name='control_tran_scan')
    # switch batch, tlen dimension back.
    return tf.transpose(next_state, [1, 0, 2])

  def deep_smooth(self, obs_tensor, intervention_tensor, mask_tensor):
    """Similar to KF smooth, derive state posterior based on observation."""
    tf.logging.info('deep_smooth')
    states = self.inference_network(obs_tensor, intervention_tensor,
                                    mask_tensor, self._config.infer_model)
    tf.logging.info('states')
    return tuple(states)

  def _encode(self, tensor_dict: Dict[bytes, tf.Tensor]) -> tf.Tensor:
    """Encode an input tensor based on DKF model.

    Args:
      tensor_dict: A dict of tensor.

    Returns:
      dict of network output tensor.
    """
    train_and_predict_tensors = self._get_train_and_predict_tensors(
        tensor_dict, self._config.sys_id_len)
    obs_train_tensor, obs_train_mask_tensor, intervention_train_tensor, obs_to_trigger_tensor, obs_to_trigger_mask_tensor, intervention_to_trigger_tensor, biomarker_boolean_mask_tensor = train_and_predict_tensors  # pylint:

    batch_size = obs_train_tensor.get_shape().as_list()[0]

    states = self.deep_smooth(obs_train_tensor, intervention_train_tensor,
                              obs_train_mask_tensor)

    state_for_prediction = self.deep_smooth(obs_to_trigger_tensor,
                                            intervention_to_trigger_tensor,
                                            obs_to_trigger_mask_tensor)

    # mu_smooth shape [bs, tlen, _z_dim]
    mu_smooth = states[0]
    mu_prediction = state_for_prediction[0]

    # mu_smooth shape [bs, tlen, _z_dim, _z_dim]
    sigma_smooth = states[1]
    sigma_prediction = state_for_prediction[1]

    # Sample from smoothing distribution
    if self._config.use_jitter:
      jitter = 1e-2 * tf.eye(
          tf.shape(sigma_smooth)[-1], batch_shape=tf.shape(sigma_smooth)[0:-2])
      mvn_smooth = tfp.distributions.MultivariateNormalTriL(
          mu_smooth, sigma_smooth + jitter)
    else:
      mvn_smooth = tfp.distributions.MultivariateNormalTriL(
          mu_smooth, sigma_smooth)

    # Note the following method is not stable on cholesky op.
    # mvn_smooth = MultivariateNormalTriL(mu_smooth, tf.cholesky(Sigma_smooth))
    # z_smooth shape [bs, tlen, _z_dim];
    z_smooth = mvn_smooth.sample()

    # Transition distribution \prod_{t=2}^T p(z_t|z_{t-1}, u_{t})
    # We use tm1 to denote t-1;
    # state_tran_z_tm1 to denote state_tran(z_{t-1}).
    # control_tran_u_t to denote control_tran(u_t).
    # We need to evaluate N(z_t; state_tran_z_tm1 + control_tran_u_t, Q)
    # Roll left to remove the first input
    # intervention_tensor: [bs, tlen, _u_dim]
    z_tm1 = z_smooth[:, :-1, :]
    u_t = intervention_train_tensor[:, 1:, :]
    tf.logging.info(u_t)
    # mu_transition shape [bs * (tlen - 1), _z_dim]
    mu_transition = tf.reshape(
        self.state_tran(z_tm1) + self.control_tran(u_t), [-1, self._z_dim])

    # z_t_transition [bs * (tlen - 1), _z_dim]
    z_t_transition = tf.reshape(z_smooth[:, 1:, :], [-1, self._z_dim])

    # We transform the rand var to be zero-mean:
    # N(z_t; Az_tm1 + Bu_t, Q) as N(z_t - Az_tm1 - Bu_t; 0, Q)
    trans_centered = z_t_transition - mu_transition
    # mvn_transition [bs * (tlen - 1), self._z_dim]
    mvn_transition = MultivariateNormalTriL(
        tf.zeros(self._z_dim), tf.cholesky(self.state_noise))
    # log_prob_transition [bs * (tlen - 1)]
    log_prob_transition = mvn_transition.log_prob(trans_centered)

    ## Emission distribution \prod_{t=1}^T p(obs_t|z_t)
    # We need to evaluate N(y_t; Cz_t, R). We write it as N(y_t - Cz_t; 0, R)
    # z_smooth shape [bs, tlen, z_dim];
    # self.obs_emission shape [a_dim, z_dim];
    # obs_emission_z_t shape [bs, tlen, _a_dim]
    obs_emission_z_t = self.obs_emission(z_smooth)
    obs_emission_z_t_resh = tf.reshape(obs_emission_z_t,
                                       [-1, self._out_obs_dim])

    # observation tensor reshaped.
    tf.logging.info(biomarker_boolean_mask_tensor)  # [num_obs]
    tf.logging.info(obs_train_tensor)  # [bs, tlen, num_obs]
    y_t_resh = tf.reshape(
        tf.transpose(
            tf.boolean_mask(
                tf.transpose(obs_train_tensor, [2, 0, 1]),
                biomarker_boolean_mask_tensor), [1, 2, 0]),
        [-1, self._out_obs_dim])
    emiss_centered = y_t_resh - obs_emission_z_t_resh
    mask_flat = tf.reshape(
        tf.transpose(
            tf.boolean_mask(
                tf.transpose(obs_train_mask_tensor, [2, 0, 1]),
                biomarker_boolean_mask_tensor), [1, 2, 0]),
        [-1, self._out_obs_dim])
    # set missing obs emission center to be zero.
    # emiss_centered shape [bs * tlen, _a_dim]
    emiss_centered = tf.multiply(mask_flat, emiss_centered)

    mvn_emission = MultivariateNormalTriL(
        tf.zeros(self._out_obs_dim), tf.cholesky(self.obs_noise))

    # log_prob_emission shape [bs * tlen].
    log_prob_emission = mvn_emission.log_prob(emiss_centered)

    if self._config.pretrain_interv:
      # Interv distribution \prod_{t=0}^T-1 p(interv_t+1|z_t)
      interv_forecast_z_t = self.interv_forecast(z_tm1)
      interv_forecast_z_t_resh = tf.reshape(interv_forecast_z_t,
                                            [-1, self._u_dim])
      u_t_resh = tf.reshape(u_t, [-1, self._u_dim])

      interv_centered = u_t_resh - interv_forecast_z_t_resh
      mvn_interv = MultivariateNormalTriL(
          tf.zeros(self._u_dim), tf.cholesky(self.interv_noise))

      # log_prob_interv shape [bs * tlen].
      log_prob_interv = mvn_interv.log_prob(interv_centered)

    ## Distribution of the initial state p(z_1|z_0)
    z_0 = z_smooth[:, 0, :]
    init_mu = tf.zeros([batch_size, self._z_dim])
    init_sigma = tf.reshape(
        tf.tile(
            tf.eye(self._z_dim, num_columns=self._z_dim),
            tf.constant([batch_size, 1])),
        [batch_size, self._z_dim, self._z_dim])
    mvn_0 = MultivariateNormalTriL(init_mu, tf.cholesky(init_sigma))
    log_prob_0 = mvn_0.log_prob(z_0)

    # Entropy log(\prod_{t=1}^T p(z_t|y_{1:T}, u_{1:T}))
    entropy = -mvn_smooth.log_prob(z_smooth)
    entropy = tf.reshape(entropy, [-1])
    # entropy = tf.zeros(())

    log_probs = [
        tf.reduce_mean(log_prob_transition),
        tf.reduce_mean(log_prob_emission),
        tf.reduce_mean(log_prob_0),
        tf.reduce_mean(entropy)
    ]
    if self._config.pretrain_interv:
      log_probs = log_probs + [tf.reduce_mean(log_prob_interv)]

    kf_elbo = tf.reduce_sum(log_probs)

    state_loss = [
        tf.reduce_mean(log_prob_transition),
        tf.reduce_mean(log_prob_0),
        tf.reduce_mean(entropy)
    ]
    state_only_loss = tf.reduce_sum(state_loss)

    output = dict()
    # loss and obs prediction.
    if self._config.sys_id_len > 0:
      tlen = self._config.sys_id_len
    else:
      tlen = self._config.context_window_size
    # obs_est starting from t=2
    # obs_est only for output prediction, not used for loss computation.
    output['obs_est'] = tf.reshape(obs_emission_z_t,
                                   [-1, tlen, self._out_obs_dim])

    # mu_smooth shape [bs, tlen, z_dim];
    # final state_encoding shape [bs, z_dim]
    output['state_encoding'] = mu_prediction[:, -1, :]

    # final state_traj_encoding shape [bs, tlen, z_dim]
    output['state_traj_encoding'] = mu_prediction[:, :, :]

    # full_state_encoding carries mu_smooth[:, -1, :] and
    # sigma_smooth [:, -1, :, :] to reconstruct the full distribution.
    # Its shape is [bs, z_dim, z_dim + 1]
    output['full_state_encoding'] = tf.concat([
        tf.expand_dims(mu_prediction[:, -1, :], axis=-1),
        sigma_prediction[:, -1, :, :]
    ],
                                              axis=2)
    if self._config.state_only_loss:
      output['loss'] = -state_only_loss
    else:
      output['loss'] = -kf_elbo
    # output['last_obs']  shape [bs, _out_obs_dim]
    output['last_obs'] = tf.squeeze(
        tf.slice(obs_to_trigger_tensor,
                 [0, self._config.context_len_to_trigger - 1, 0], [-1, 1, -1]))
    if self._config.forecast_biomarkers:
      # switch shape to [_out_obs_dim, bs] for applying mask.
      output['last_obs'] = tf.boolean_mask(
          tf.transpose(output['last_obs']), biomarker_boolean_mask_tensor)
      # transpose shape back.
      output['last_obs'] = tf.transpose(output['last_obs'])

    output['state_loss'] = state_loss

    tf.logging.info(output)
    return output


def build_model_ops(features: Dict[bytes, tf.Tensor],
                    labels: Dict[bytes, tf.Tensor],
                    mode: tf.contrib.learn.ModeKeys,
                    params: experiment_config_pb2.ModelConfig = None):
  """Create a model for use with a tf.contrib.learn.Estimator.

  Args:
    features: Dictionary with keys as feature codes, values representing feature
      tensor with shape [batch_size, context_window_size].
    labels: Dictionary with keys as label names, values representing tensor with
      shape [batch_size].
    mode: One of {TRAIN,EVAL}.
    params: ModelConfig that specifies the
        * model_name: .
        * labels: a list of Label objects.

  Returns:
    An instance of tf.contrib.learn.ModelFnOps.
  """
  model = _build_model(params.model_name, mode, config=params)
  network_output = model.generate_output(features)
  output_heads = []
  logits_dict = {}
  logits_dimension = params.logits_dimension
  weights = []

  if params.is_dynamic_system_model:
    head = sequence_heads.ObservationHead(
        name='system_id', model_hparams=params)
    output_heads.append(head)
    # loss
    logits_dict['system_id'] = network_output['obs_est']
    if 'loss' in network_output.keys():
      # loss is passed to network output in KF encoder.
      logits_dict['system_id'] = (logits_dict['system_id'],
                                  network_output['loss'])
    if labels is not None:
      labels['system_id'] = features
    weights.append(params.sys_id_weight)

  for label in params.labels:
    if params.is_survival_model:
      if params.model_name == 'mri':
        logits_dimension = network_output['state_encoding'].get_shape()[1]
      if label.survival_model_name == 'correlated':
        # logits_dimension for each head is the same as number of events
        # for correlated events, as all event rates will be used to compute
        # loss and prediction. The logits for each head are transformed from
        # raw network output logits, which have the same dimension as number of
        # relations.
        logits_dimension = len(params.labels)
        head = survival_heads.SurvivalHead(
            name=label.name,
            survival_model_name=label.survival_model_name,
            label_dimension=logits_dimension,
            model_hparams=params,
            all_event_list=[l.name for l in params.labels])
      else:
        head = survival_heads.SurvivalHead(
            name=label.name,
            survival_model_name=label.survival_model_name,
            label_dimension=logits_dimension,
            model_hparams=params,
            all_event_list=None)
    elif label.num_classes == 2:
      head = tf.contrib.estimator.binary_classification_head(name=label.name)
    else:
      head = tf.contrib.estimator.multi_class_head(
          n_classes=label.num_classes, name=label.name)
      logits_dimension = label.num_classes
    output_heads.append(head)

    if params.model_name == 'mri' and params.is_survival_model:
      logits = network_output['state_encoding']
    else:
      logits = tf.layers.dense(
          network_output['state_encoding'],
          units=logits_dimension,
          activation=None)
    logits_dict[label.name] = logits
    weights.append(label.weight)

  # collect all_heads.
  if params.is_survival_model:
    all_heads = multi_head.SurvivalMultiHead(
        heads=output_heads, head_weights=weights)
  else:
    all_heads = tf.contrib.estimator.multi_head(output_heads, weights)

  def train_op_fn(loss):
    optimizer = _get_optimizer(
        params.optimizer.name,
        learning_rate=params.learning_rate,
        momentum=params.optimizer.momentum)

    return slim.learning.create_train_op(loss, optimizer)

  return all_heads.create_estimator_spec(
      features=features,
      mode=mode,
      logits=logits_dict,
      labels=labels,
      train_op_fn=train_op_fn)


MODEL_MAP = {
    'mlp': MultiLayerPerceptron,
    'mri': MostRecentInput,
    'lstm_ds': LSTMDynamicSystem,
    'dkf_ds': DKFDynamicSystem,
}


def _build_model(name: bytes,
                 mode: tf.contrib.learn.ModeKeys,
                 config: experiment_config_pb2.ModelConfig = None):
  """Constructs a model by name.

  Args:
    name: Model name.
    mode: One of {TRAIN,EVAL}.
    config: Model config.

  Returns:
    An instance of the Encoder class, created with `config`.
  """
  try:
    model_obj = MODEL_MAP[name.lower()]
    return model_obj(mode, config=config)
  except KeyError:
    raise ValueError('Unrecognized model name %s' % name)


def _get_optimizer(name, learning_rate, momentum=None):
  """Constructs an optimizer by name.

  Args:
    name: Optimizer name.
    learning_rate: learning rate.
    momentum: Default to None. Only used when optimizer name is 'momentum' or
      'rmsprop'

  Returns:
    An optimizer with the given name and learning rate.
  """
  optimizer_name = name.lower()
  if optimizer_name == 'sgd':
    return tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_name == 'momentum':
    return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    return tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
  elif optimizer_name == 'adam':
    return tf.train.AdamOptimizer(learning_rate)
  elif optimizer_name == 'adagrad':
    return tf.train.AdagradOptimizer(learning_rate)
  else:
    raise ValueError('Unrecognized optimizer "%s"!' % name)
