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

"""Customized layers to be used by unsupervised learning of patient representation."""

import tensorflow as tf


class MLPLayer(tf.keras.layers.Layer):
  """A utility layer for multiple dense layer MLP."""

  def __init__(self, num_layer, num_hidden_unit, num_output_unit,
               output_activation):
    super(MLPLayer, self).__init__()
    self.num_layer = num_layer
    self.num_hidden_unit = num_hidden_unit
    self.num_output_unit = num_output_unit
    self.output_activation = output_activation

  def build(self, input_shape):
    self.hidden_layers = dict()
    for layer in range(self.num_layer - 1):
      self.hidden_layers[layer] = tf.keras.layers.Dense(
          self.num_hidden_unit, input_shape=input_shape, activation='relu')
    self.out_layer = tf.keras.layers.Dense(
        self.num_output_unit, activation=self.output_activation)

  def call(self, inputs):
    x = inputs
    for layer in range(self.num_layer - 1):
      x = self.hidden_layers[layer](x)
    x = self.out_layer(x)
    return x


class AttentionLayer(tf.keras.layers.Layer):
  """A layer that assigns normalized attention weights to inputs."""

  def __init__(self,
               num_predict,
               normalization_model='softmax',
               attention_init_value=None):
    super(AttentionLayer, self).__init__()
    self.num_predict = num_predict
    self.attention_init_value = attention_init_value
    self.normalization_model = normalization_model

  def build(self, input_shape):
    # kernel with shape [num_feature]
    if self.attention_init_value is not None:
      val_str_list = self.attention_init_value.split(',')
      val_list = [float(val_str) for val_str in val_str_list]
      self.kernel = self.add_weight(
          'kernel',
          shape=[int(input_shape[2])],
          initializer=tf.keras.initializers.Constant(
              value=tf.constant(val_list)))
    else:
      self.kernel = self.add_weight(
          'kernel',
          shape=[int(input_shape[2])],
          initializer='uniform',
          dtype='float32')
    super(AttentionLayer, self).build(input_shape)

  def call(self, inputs):
    # softmax is taken along feature dim.
    if self.normalization_model == 'softmax':
      self.attention_tensor = tf.nn.softmax(self.kernel)
    if self.normalization_model == 'linear':
      positive_kernel = tf.math.softplus(self.kernel)
      kernel_sum = tf.reduce_sum(positive_kernel)
      self.attention_tensor = tf.math.divide_no_nan(positive_kernel, kernel_sum)

    # attention_tensor shape [batch_size, num_predict, num_feature]
    self.attention_tensor = tf.repeat(
        tf.expand_dims(self.attention_tensor, axis=0),
        repeats=[self.num_predict],
        axis=0)
    return tf.multiply(inputs, self.attention_tensor)

  @property
  def attention_outputs(self):
    return self.attention_tensor
