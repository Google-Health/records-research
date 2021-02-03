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

"""Encoders and decoders to be used by unsupervised learning of patient representation."""

import tensorflow as tf

import customized_layers


class LSTMDecoder(tf.keras.layers.Layer):
  """Using a LSTM for decoding and output a sequence of predictions."""

  def __init__(self, **kwargs):
    super().__init__()
    self.num_feature = kwargs.pop('num_feature')
    self.num_predict = kwargs.pop('num_predict')
    self.state_size = kwargs.pop('state_size')
    self.lstm_layer = tf.keras.layers.LSTM(
        units=self.state_size, return_state=True, name='decoder')
    self.obs_emission = tf.keras.layers.Dense(
        self.num_feature, input_shape=(self.state_size,))

  def lstm_one_step(self, previous_output, previous_state):
    current_output, current_state_h, current_state_c = self.lstm_layer(
        tf.expand_dims(previous_output, axis=1), initial_state=previous_state)

    return current_output, [current_state_h, current_state_c]

  def state_tran_step_fn(self, previous_tuple, dummy_input_elem):
    del dummy_input_elem
    previous_output, previous_state = previous_tuple
    current_output, current_state = self.lstm_one_step(previous_output,
                                                       previous_state)

    return (current_output, current_state)

  def forecast_value(self, inputs):
    dummy_input_sequence = tf.zeros([self.num_predict])
    # [state_h, state_c] = state
    forecast_output, _ = tf.scan(
        self.state_tran_step_fn,
        elems=dummy_input_sequence,  # just control the forecast length.
        initializer=inputs,  # output, state from encoder
        parallel_iterations=10,
        name='obs_forecast_scan')
    # switch batch, tlen dimension back.
    # forecast_output shape [batch_size, num_predict, num_feature]
    forecast_output = tf.transpose(forecast_output, [1, 0, 2])
    forecast_obs = self.obs_emission(forecast_output)
    # remove the last prediction from forecast_obs.
    return forecast_obs

  def call(self, inputs):
    forecast_obs = self.forecast_value(inputs)
    # forecast_obs.shape (None, predict_horizon, num_feature)
    return forecast_obs


class MLPDecoder(tf.keras.layers.Layer):
  """MLP decoder for binary classification with a single dense layer ."""

  def __init__(self, **kwargs):
    super().__init__()
    self.out = customized_layers.MLPLayer(
        num_layer=1,
        num_output_unit=1,
        num_hidden_unit=None,
        output_activation='sigmoid')

  def call(self, inputs):
    output, _ = inputs
    output = self.out(output)
    return output


class LSTMEncoder(tf.keras.layers.Layer):
  """LSTM encoder."""

  def __init__(self, **kwargs):
    super().__init__()
    self.state_size = kwargs.pop('state_size')
    self.hidden_lstm = tf.keras.layers.LSTM(
        self.state_size, return_state=True, name='LSTM_encoder')

  def call(self, x):
    output, state_h, state_c = self.hidden_lstm(x)
    encoder_state = [state_h, state_c]

    return output, encoder_state

