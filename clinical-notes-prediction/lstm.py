"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
from __future__ import print_function

import g_ops
import tensorflow as tf


class LSTM:
    """Implements a Tensorflow LSTM with various dropout options."""

    class _ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
        """
        Zoneout randomly preserves the LSTM hidden state across timesteps.
        """

        def __init__(self, cell, zoneout_drop_prob, is_training=True):
            self._cell = cell
            self._zoneout_prob = zoneout_drop_prob
            self._is_training = is_training

        @property
        def state_size(self):
            return self._cell.state_size

        @property
        def output_size(self):
            return self._cell.output_size

        def __call__(self, inputs, state, scope=None):
            output, new_state = self._cell(inputs, state, scope=scope)
            final_new_state = [None, None]
            if self._is_training:
                for i, state_element in enumerate(state):
                    random_tensor = 1 - self._zoneout_prob
                    random_tensor += tf.random_uniform(tf.shape(state_element))
                    binary_tensor = tf.floor(random_tensor)
                    final_new_state[i] = (new_state[i] - state_element
                            ) * binary_tensor + state_element
            else:
                for i, state_element in enumerate(state):
                    final_new_state[i] = state_element * self._zoneout_prob + (
                            new_state[i] * (1 - self._zoneout_prob))

            returned_state = tf.nn.rnn_cell.LSTMStateTuple(
                    final_new_state[0], final_new_state[1])

            return output, returned_state

    def __init__(self,
                 model_dim,
                 bidirectional=False,
                 input_dropout_keep_prob=1.0,
                 hidden_dropout_keep_prob=1.0,
                 variational_input_keep_prob=1.0,
                 variational_output_keep_prob=1.0,
                 zoneout_keep_prob=1.0,
                 is_training=True,
                 trainable=True):
        """Params:

        model_dim: dimension of the LSTM hidden state.
        bidirectional: whether to use a bidirectional LSTM.
        input_dropout_keep_prob: dropout rate for model inputs.
        hidden_dropout_keep_prob: dropout rate for model hidden state.
        variational_input_keep_prob: variational dropout rate for model inputs.
        variational_output_keep_prob: variational dropout rate for model hidden
          state.
        zoneout_keep_prob: rate to apply Zoneout across timesteps.
        is_training: whether model is in training or eval mode.
        trainable: whether LSTM weights should be updated during training.
        """
        self._is_training = is_training
        self._trainable = trainable
        self._model_dim = model_dim
        self._bidirectional = bidirectional
        self._input_dropout_keep_prob = input_dropout_keep_prob
        self._hidden_dropout_keep_prob = hidden_dropout_keep_prob
        self._variational_input_keep_prob = variational_input_keep_prob
        self._variational_output_keep_prob = variational_output_keep_prob
        self._zoneout_keep_prob = zoneout_keep_prob

    def _LSTMCell(self):
        """Initializes the LSTM cell with dropout and Zoneout applied."""
        cell = tf.nn.rnn_cell.LSTMCell(self._model_dim, trainable=self._trainable)
        if self._zoneout_keep_prob < 1.0:
            cell = self._ZoneoutWrapper(
                    cell, 1.0 - self._zoneout_keep_prob, self._is_training)
        if self._is_training:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    input_keep_prob=self._input_dropout_keep_prob,
                    output_keep_prob=self._hidden_dropout_keep_prob)
        return cell

    def _VariationalDropout(self, cell, input_size):
        """Adds variational dropout to the LSTM cell.

        Applied separately from _LSTMCell, as this wrapper depends on input depth.
        """
        return tf.nn.rnn_cell.DropoutWrapper(
                cell,
                input_keep_prob=self._variational_input_keep_prob,
                output_keep_prob=self._variational_output_keep_prob,
                variational_recurrent=True,
                input_size=input_size)

    def Forward(self, inputs, sequence_length):
        """Constructs the forward graph and returns the LSTM output tensors."""
        fw_cell = self._LSTMCell()
        if self._bidirectional:
            bw_cell = self._LSTMCell()

        if self._is_training:
            input_size = inputs.shape[2]
            fw_cell = self._VariationalDropout(fw_cell, input_size)
            if self._bidirectional:
                bw_cell = self._VariationalDropout(bw_cell, input_size)

        if self._bidirectional:
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    inputs,
                    sequence_length=sequence_length)
            final_state_fw, final_state_bw = final_state
            last_output_fw = final_state_fw[-1].h
            last_output_bw = final_state_bw[-1].h
            outputs = tf.concat(outputs, 2)
            last_outputs = tf.concat([last_output_fw, last_output_bw], 1)
        else:
            outputs, final_state = tf.nn.dynamic_rnn(
                    fw_cell,
                    inputs,
                    sequence_length=sequence_length)
            last_outputs = final_state[-1].h

        return outputs, last_outputs

