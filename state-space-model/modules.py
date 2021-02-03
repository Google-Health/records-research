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
import tensorflow as tf


def _recent_input_module(input_tensor: tf.Tensor) -> tf.Tensor:
  """Create a module of that takes the most recent input.

  Args:
    input_tensor: input tensor of shape [batch_size, window_size, num_features].

  Returns:
    A Tensor of shape [batch_size, num_features].
  """
  return input_tensor[:, -1, :]


def _fully_connected_module(input_tensor: tf.Tensor, num_dense_layers: int,
                            dense_layer_size: int, output_layer_size: int,
                            output_activation, is_training: bool,
                            drop_rate: float,
                            name: bytes,
                            trainable: bool = True) -> tf.Tensor:
  """Create a module of multiple connected dense layers.

  Args:
    input_tensor: input tensor of shape [batch_size, input_dimension].
    num_dense_layers: number of dense layers in this module.
    dense_layer_size: number of hidden units for each dense layer.
    output_layer_size: number of units for output layer.
    output_activation: output layer activation function.
    is_training: whether being used in training mode.
    drop_rate: dropout probability. Dropout is applied to each dense layer under
      training mode.
    name: module name.
    trainable: whether vars in this module are trainable.

  Returns:
    A Tensor of shape [batch_size, output_dimension].
  """
  net_output = input_tensor
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    for layer_num in range(num_dense_layers - 1):
      net_output = tf.layers.dense(
          net_output,
          dense_layer_size,
          activation=tf.nn.relu,
          trainable=trainable,
          name=name + '_' + str(layer_num))

    net_output = tf.layers.dense(
        net_output,
        output_layer_size,
        activation=output_activation,
        trainable=trainable,
        name=name + '_' + str(num_dense_layers - 1))

    net_output = tf.layers.dropout(
        net_output, rate=drop_rate, training=is_training)

  return net_output


# set name_tag = '' for reuse.
def _interv_forecast_module(state, interv_size, interv_nmlp, interv_smlp,
                            name_tag, trainable=True):
  # state shape [batch_size, state_size]
  return _fully_connected_module(
      input_tensor=state,
      num_dense_layers=interv_nmlp,
      dense_layer_size=interv_smlp,
      output_layer_size=interv_size,
      output_activation=tf.nn.sigmoid,
      is_training=True,
      drop_rate=0,
      name='interv_forecast' + name_tag,
      trainable=trainable)


def _hazard_emission_module(state, hazard_nmlp, hazard_smlp, name_tag):
  # state shape [batch_size, state_size]
  return _fully_connected_module(
      input_tensor=state,
      num_dense_layers=hazard_nmlp,
      dense_layer_size=hazard_smlp,
      output_layer_size=1,
      output_activation=tf.nn.sigmoid,
      is_training=True,
      drop_rate=0,
      name='hazard_emission' + name_tag,
      trainable=True)


def _obs_emission_module(state, obs_size, obs_nmlp, obs_smlp, name_tag,
                         emission_activation):
  # state shape [batch_size, state_size]
  return _fully_connected_module(
      input_tensor=state,
      num_dense_layers=obs_nmlp,
      dense_layer_size=obs_smlp,
      output_layer_size=obs_size,
      output_activation=emission_activation,
      is_training=True,
      drop_rate=0,
      name='obs_emission' + name_tag,
      trainable=True)


def _state_tran_module(state, state_size, stran_nmlp, stran_smlp, name_tag,
                       trainable=True):
  """Create a module for state transition.

  Args:
    state: input tensor of shape [batch_size, state_size].
    state_size: size of output layer.
    stran_nmlp: number of layers.
    stran_smlp: number of hidden units for each dense layer.
    name_tag: name tag to differentiate different.
    trainable: whether vars in this module are trainable.

  Returns:
    state_mean: mean value tensor of state shape [batch_size, state_size].
  """

  # state shape [batch_size, state_size]
  tf.logging.info(state)
  gate = _fully_connected_module(
      input_tensor=state,
      num_dense_layers=stran_nmlp,
      dense_layer_size=stran_smlp,
      output_layer_size=state_size,
      output_activation=tf.nn.sigmoid,
      is_training=True,
      drop_rate=0,
      name='state_tran_gate' + name_tag,
      trainable=trainable)
  nonlinear_state_candidate = _fully_connected_module(
      input_tensor=state,
      num_dense_layers=stran_nmlp,
      dense_layer_size=stran_smlp,
      output_layer_size=state_size,
      output_activation=None,
      is_training=True,
      drop_rate=0,
      name='nonlinear_state_candidate' + name_tag,
      trainable=trainable)
  linear_state_candidate = _fully_connected_module(
      input_tensor=state,
      num_dense_layers=1,
      dense_layer_size=stran_smlp,
      output_layer_size=state_size,
      output_activation=None,
      is_training=True,
      drop_rate=0,
      name='linear_state_candidate' + name_tag,
      trainable=trainable)

  state_mean = tf.multiply(tf.ones_like(gate) - gate,
                           linear_state_candidate) + tf.multiply(
                               gate, nonlinear_state_candidate)
  return state_mean


def _control_tran_module(interv, state_size, ctran_nmlp, ctran_smlp, name_tag,
                         trainable=True):
  """Create a module for control transition.

  Args:
    interv: input tensor of shape [batch_size, interv_size].
    state_size: size of output layer.
    ctran_nmlp: number of layers.
    ctran_smlp: number of hidden units for each dense layer.
    name_tag: name tag to differentiate different.
    trainable: whether vars in this module are trainable.

  Returns:
    state_mean: mean value tensor of state shape [batch_size, state_size].
  """

  control_gate = _fully_connected_module(
      input_tensor=interv,
      num_dense_layers=ctran_nmlp,
      dense_layer_size=ctran_smlp,
      output_layer_size=state_size,
      output_activation=tf.nn.sigmoid,
      is_training=True,
      drop_rate=0,
      name='control_tran_gate' + name_tag,
      trainable=trainable)
  control_nonlinear_state_candidate = _fully_connected_module(
      input_tensor=interv,
      num_dense_layers=ctran_nmlp,
      dense_layer_size=ctran_smlp,
      output_layer_size=state_size,
      output_activation=None,
      is_training=True,
      drop_rate=0,
      name='control_nonlinear_state_candidate' + name_tag,
      trainable=trainable)
  control_linear_state_candidate = _fully_connected_module(
      input_tensor=interv,
      num_dense_layers=1,
      dense_layer_size=ctran_smlp,
      output_layer_size=state_size,
      output_activation=None,
      is_training=True,
      drop_rate=0,
      name='control_linear_state_candidate' + name_tag,
      trainable=trainable)

  state_mean = tf.multiply(
      tf.ones_like(control_gate) - control_gate,
      control_linear_state_candidate) + tf.multiply(
          control_gate, control_nonlinear_state_candidate)
  return state_mean
