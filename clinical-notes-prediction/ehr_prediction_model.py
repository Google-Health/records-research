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

from input_transformer import InputTransformer
from lstm import LSTM


class EHRPredictionModel:
    """Top-level modeling code for generating predictions and losses."""

    def __init__(self,
                 input_transformer_params,
                 record_model_params,
                 num_classes=1,
                 model_type='multilabel',
                 pretrain_num_sampled=512,
                 pretrain_freq_file=''):
        """Params:

        input_transformer_params: dict of params for InputTransformer.
        record_model_params: dict of params for the record-level LSTM.
        num_classes: number of classes to predict (1 for binary).
        model_type: 'multilabel' or 'multiclass'. Use 'multilabel' if binary.
        pretrain_num_sampled: number to sample for the pretraining loss.
        pretrain_freq_file: vocab frequency file to use for the sampling
          distribution during pretraining.
        """
        self._input_transformer = InputTransformer(**input_transformer_params)
        self._record_model = LSTM(**record_model_params)
        self._num_classes = num_classes
        self._model_type = model_type
        self._pretrain_num_sampled = pretrain_num_sampled
        self._pretrain_freq_file = pretrain_freq_file
        self._notes_feature_name = input_transformer_params['notes_feature_name']
        self._hierarchical = input_transformer_params['use_notes_model']
        if self._hierarchical:
            self._notes_model_dim = input_transformer_params[
                    'notes_model']['model_dim'])
            self._notes_bidirectional = input_transformer_params[
                    'notes_model']['bidirectional'])

    def Prediction(self,
                   sequence_features,
                   dense_features,
                   sequence_lengths,
                   delta_time):
        """Generates the logits from the input features."""
        transformed_inputs = self._input_transformer.ProcessInputs(
                sequence_features, sequence_length, delta_time, dense_features)
        processed_sequence_length = transformed_inputs['seq_length']
        record_model_inputs = transformed_inputs['seq_tensor']
        if self._hierarchical:
            pretraining_inputs = transformed_inputs['pretrain_tensor']
        else:
            pretraining_inputs = None

        _, record_model_outputs = self._record_model.Forward(
                record_model_inputs, processed_sequence_length)
        logits = tf.layers.dense(record_model_outputs, self._num_classes)
        return logits, pretraining_inputs

    def Loss(self, logits, labels):
        """Generates the loss for training."""
        if self._model_type == 'multiclass':
            return tf.losses.softmax_cross_entropy(labels, logits)
        else:
            return tf.losses.sigmoid_cross_entropy(labels, logits)

    def PretrainLoss(self, pretraining_inputs, sequence_features):
        """Generates the loss for pretraining the notes model."""
        notes_feature = sequence_features[self._notes_feature_name]
        word_tokens = tf.expand_dims(notes_feature.values, axis=1)
        word_indices = notes_feature.indices
        batched_tokens, notes_length, _ = g_ops.batch_by_index_pair(
                word_tokens, tf.to_int32(word_indices))
        batched_tokens = tf.squeeze(batched_tokens, axis=2)

        # The next-step pretraining targets are words 1:n in each note.
        # In the bidirectional case, the last-step targets are words 0:n-1.
        fw_mask = tf.sequence_mask(notes_length, maxlen=tf.shape(batched_tokens)[1])
        if self._notes_bidirectional:
            bw_mask = tf.sequence_mask(
                    notes_length - 1, maxlen=tf.shape(batched_tokens)[1])
            fw_targets = tf.boolean_mask(batched_tokens[:, 1:], fw_mask[:, 1:])
            bw_targets = tf.boolean_mask(batched_tokens, bw_mask)
            notes_targets = tf.concat([fw_targets, bw_targets], axis=0)
        else:
            notes_targets = tf.boolean_mask(batched_tokens[:, 1:], fw_mask[:, 1:])

        vocab_size = self._input_transformer.VocabSize(self._notes_feature_name)
        if self._pretrain_freq_file:
            sampler = tf.nn.fixed_unigram_candidate_sampler(
                    notes_targets,
                    num_true=1,
                    num_sampled=self._pretrain_num_sampled,
                    unique=True,
                    range_max=vocab_size,
                    vocab_file=self._pretrain_freq_file)
        else:
            sampler = tf.random.log_uniform_candidate_sampler(
                    notes_targets,
                    num_true=1,
                    num_sampled=self._pretrain_num_sampled,
                    unique=True,
                    range_max=vocab_size)

        pretrain_weights = tf.get_variable(
                'pretrain_weights', shape=[vocab_size, self._notes_model_dim])
        pretrain_biases = tf.get_variable('pretrain_biases', shape=[vocab_size])
        loss = tf.nn.nce_loss(
                weights=pretrain_weights,
                biases=pretrain_biases,
                labels=notes_targets,
                inputs=pretraining_inputs,
                num_sampled=self._pretrain_num_sampled,
                num_classes=vocab_size,
                sampled_values=sampler)
        return tf.reduce_mean(loss)

