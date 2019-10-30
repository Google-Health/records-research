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

from lstm import LSTM


class InputTransformer:
    """
    Given a dict of SparseTensor input features, embeds these features and
    prepares them for input to the record-level LSTM. If using a hierarchical model,
    runs the notes-level LSTM over the notes feature.
    """

    class _BaseEmbeddingLayer:
        """Base class for embedding input features."""
        def __init__(self,
                     vocab_size=-1,
                     embed_dim=-1,
                     variational_vocab_keep_prob=1.0,
                     is_training=True,
                     feature_name='',
                     embed_config_path=''):
            self._variational_vocab_keep_prob = variational_vocab_keep_prob
            self._is_training = is_training
            self._feature_name = feature_name
            if embed_config_path:
                config = g_ops.load_embedding_config_proto(embed_config_path)
                var_vocab_size = config[feature_name].vocabulary.vocabulary_size
                var_embed_dim = config[feature_name].embedding_dimension
            if vocab_size > 0:
                var_vocab_size = vocab_size
            if embed_dim > 0:
                var_embed_dim = embed_dim
            self._vocab_size = var_vocab_size
            self._embed_dim = var_embed_dim
            self._CreateVariables(self._vocab_size, self._embed_dim)

        def _CreateVariables(self, vocab_size, embed_dim):
            with tf.variable_scope(self._feature_name):
                var = tf.get_variable(name='embed_var', shape=[vocab_size, embed_dim])

            if self._is_training and self._variational_vocab_keep_prob < 1:
                random_tensor = self._variational_vocab_keep_prob
                random_tensor += tf.random_uniform([var.get_shape()[0].value, 1])
                mask_tensor = tf.floor(random_tensor) / (
                        self._variational_vocab_keep_prob)
                var *= mask_tensor

            self._vars = var

    class _DenseIdEmbeddingLayer(_BaseEmbeddingLayer):
        """Class for generating embeddings from dense Tensors."""
        def Forward(self, inputs):
            return tf.nn.embedding_lookup(self._vars, inputs)

    class _SparseIdEmbeddingLayer(_BaseEmbeddingLayer):
        """Class for generating embeddings from SparseTensors."""
        def Forward(self, inputs, rank=3):
            outputs = tf.nn.safe_embedding_lookup_sparse(self._vars, inputs)
            outputs.set_shape([None] * (rank - 1) + [self._embed_dim])
            return outputs

    def __init__(self,
                 feature_config_path,
                 dense_feature_config_path='',
                 notes_feature_name='',
                 use_notes_model=False,
                 notes_model_params=None,
                 variational_vocab_keep_prob=1.0,
                 notes_vocab_keep_prob=1.0,
                 notes_max_length=-1,
                 notes_num_splits=0,
                 bagging_timerange=3600,
                 bagging_aggregate_older_than=3600000,
                 is_training=True):
        """Params:

        feature_config_path: path to embedding config for discrete features.
        dense_feature_config_path: path to config file with statistics for
          continuous features.
        notes_feature_name: name of notes feature in inputs.
        use_notes_model: whether to run an LSTM on the notes.
        notes_model_params: dict with parameters for the notes LSTM.
        variational_vocab_keep_prob: vocabulary dropout rate for features other
          than notes.
        notes_vocab_keep_prob: vocabulary dropout rate for notes.
        notes_max_length: number of words to retain from notes per example.
        notes_num_splits: number of additional GPUs to distribute the notes
          LSTM across.
        bagging_timerange: length of timesteps to aggregate into a single bag
          (in seconds).
        bagging_aggregate_older_than: length of time before prediction beyond
          which all observations should be aggregated into a single bag (in
          seconds).
        is_training: whether model is in training or eval phase.
        """
        self._is_training = is_training
        self._feature_config_path = feature_config_path
        self._dense_feature_config_path = dense_feature_config_path
        self._notes_feature_name = notes_feature_name
        self._use_notes_model = use_notes_model
        self._variational_vocab_keep_prob = variational_vocab_keep_prob
        self._notes_vocab_keep_prob = notes_vocab_keep_prob
        self._notes_max_length = notes_max_length
        self._notes_num_splits = notes_num_splits
        self._bagging_timerange = bagging_timerange
        self._bagging_aggregate_older_than = bagging_aggregate_older_than
        self._notes_model = None
        if use_notes_model:
            self._notes_model_dim = notes_model_params['model_dim']
            self._notes_bidirectional = notes_model_params['bidirectional']
            self._notes_model = LSTM(**notes_model_params)

    def VocabSize(self, feature):
        """Returns the vocabulary size for feature."""
        feature_config = g_ops.load_embedding_config_proto(self._feature_config_path)
        return feature_config[feature].vocabulary.vocabulary_size

    def _CreateLayers(self, features_dict):
        """Initializes embedding layers for each input feature from config."""
        layers = []
        features = []
        for feature_name in sorted(features_dict):
            variational_vocab_keep_prob = self._variational_vocab_keep_prob
            if self._notes_feature_name == feature_name:
                variational_vocab_keep_prob = self._notes_vocab_keep_prob
            if self._use_notes_model and self._notes_feature_name == feature_name:
                embed_param = self._DenseIdEmbeddingLayer(
                        embed_config_path=self._feature_config_path,
                        variational_vocab_keep_prob=variational_vocab_keep_prob,
                        is_training=self._is_training,
                        feature_name=feature_name)
            else:
                embed_param = self._SparseIdEmbeddingLayer(
                        embed_config_path=self._feature_config_path,
                        variational_vocab_keep_prob=variational_vocab_keep_prob,
                        is_training=self._is_training,
                        feature_name=feature_name)
            layers.append(layer)
            features.append(feature_name)
        return layers, features

    def _Bagging(self, features_dict, bagging_feature, sequence_length):
        """Aggregate features into coarse-grained timesteps ('bags').

        Reduces effective sequence length for LSTM training.
        """
        bagging_feature = tf.squeeze(bagging_feature, [-1])
        if self._bagging_aggregate_older_than > 0:
            agg_threshold = self._bagging_aggregate_older_than // (
                    self._bagging_timerange)
            recent_mask = tf.less_equal(bagging_feature, agg_threshold)
            history_value = tf.zeros_like(bagging_feature) + agg_threshold + 1
            bagging_feature = tf.where(recent_mask, bagging_feature, history_value)

        bagging_ids, dt_unique, bagging_seq_length = g_ops.bagging_feature_to_ids(
                bagging_feature, sequence_length)
        bagged_features_dict = {}
        for feature in features_dict:
            bagged_features_dict[feature] = g_ops.bagging(
                    features_dict[feature], bagging_ids, bagging_seq_length)

        return bagged_features_dict, bagging_ids, dt_unique, bagging_seq_length

    def _Attention(inputs, seq_lengths):
        """Compute an attention-weighted sum over inputs."""
        batch_size, max_seq_len, embedding_dim = tf.shape(inputs)
        att_logits = tf.layers.dense(inputs, units=1)
        prior_attention_logit = tf.get_variable(
                'prior_attention_logit', shape=[1, 1, 1])
        prior_attention_logit = tf.tile(
                prior_attention_logit, multiples=[batch_size, 1, 1])
        att_logits = tf.concat([prior_attention_logit, att_logits], axis=1)
        attention_weights = g_ops.sequence_softmax(att_logits, seq_lengths + 1)
        seq_weights = attention_weights[:, 1:, :]
        embedding_sum = tf.reduce_sum(inputs * seq_weights, axis=1)
        prior = tf.get_variable('prior_embedding', shape=[1, embedding_dim])
        prior_weights = tf.reshape(weights[:, :1, :], [batch_size, 1])
        return embedding_sum + prior_weights * prior

    def ProcessInputs(self, seq_features, sequence_length, delta_time,
                      dense_features=None):
        """Prepare inputs for the record-level LSTM.

        Discrete features are embedded, and continuous features are standardized.
        If using a hierarchical model, an LSTM is run over the notes feature.
        All processed features are concatenated into a single batch of sequences.
        """
        if self._notes_feature_name and self._notes_max_length > 0:
            seq_features[self._notes_feature_name] = g_ops.sparse_nd_select_last_k(
                    seq_features[self._notes_feature_name],
                    self._notes_max_length)
        delta_time = tf.to_int64(tf.floor_div(delta_time, self._bagging_timerange))
        processed_seq_features, bagging_ids, _, processed_seq_length = self._Bagging(
           seq_features, delta_time, sequence_length)
        processed_dense_features = None
        if dense_features:
            processed_dense_features, _, dt_unique, _ = self._Bagging(
                    dense_features, delta_time, sequence_length)

        seq_outputs = []
        embedding_layers, features = self._CreateLayers(processed_seq_features)
        for layer, feature in zip(embedding_layers, features):
            # Apply an RNN on the notes feature to generate embeddings.
            if self._use_notes_model and self._notes_feature_name == feature:
                notes_feature = seq_features[feature]
                # 2-D tensor with shape [num_tokens, embed_dim].
                notes = layer.Forward(notes_feature.values)
                # 2-D tensor with shape [N, 3], where the values for second dim are
                # (batch_index, time_index, token_index).
                notes_indices = notes_feature.indices
                # Batch notes for the notes model.
                (batched_notes, notes_seq_length, notes_orig_indices
                        ) = g_ops.batch_by_index_pair(
                                notes, tf.to_int32(notes_indices))
                batched_notes.set_shape([None, None, notes.shape[-1]])
                batch_max_length = tf.shape(batched_notes)[1]
                # Because the notes batch can be very large, distribute across
                # multiple GPUs if available.
                # Place the notes on GPUs 1:notes_num_splits, with GPU 0 reserved for
                #   the record model.
                if self._notes_num_splits > 0:
                    size_per_split = tf.shape(batched_notes)[0] // (
                            self._notes_num_splits)
                    remainder = tf.shape(batched_notes)[0] % self._notes_num_splits
                    split_sizes = [size_per_split] * self._notes_num_splits
                    split_sizes[0] += remainder
                    split_notes = tf.split(
                            batched_notes, num_or_size_splits=split_sizes)
                    split_lengths = tf.split(
                            notes_seq_length, num_or_size_splits=split_sizes)
                    device_list = range(1, self._notes_num_splits + 1)
                    split_batch = zip(split_notes, split_lengths, device_list)
                else:
                    split_batch = [(batched_notes, notes_seq_length, 0)]
                model_pretrain_outputs, notes_embeddings = [], []
                with tf.variable_scope('notes_model', reuse=tf.AUTO_REUSE):
                    for split, lengths, device in split_batch:
                        with tf.device('/gpu:{}'.format(device)):
                            split_outputs, split_final_state = (
                                    self._notes_model.Forward(split, lengths)
                            # We take the first n-1 outputs for next-step prediction
                            # during pretraining. If bidirectional, we also take the
                            # last n-1 outputs from the backward LSTM for last-step
                            # prediction.
                            pretrain_outputs_mask = tf.sequence_mask(
                                    lengths - 1, maxlen=batch_max_length)
                            if self._notes_bidirectional:
                                fw_outputs, bw_outputs = tf.split(
                                        split_outputs, 2, axis=2)
                                bw_outputs = tf.manip.roll(
                                        bw_outputs, shift=-1, axis=1)
                                fw_pretrain_outputs = tf.boolean_mask(
                                        fw_outputs, pretrain_outputs_mask)
                                bw_pretrain_outputs = tf.boolean_mask(
                                        bw_outputs, pretrain_outputs_mask)
                                split_pretrain_outputs = (fw_pretrain_outputs,
                                                          bw_pretrain_outputs)
                            else:
                                split_pretrain_outputs = tf.boolean_mask(
                                        split_outputs, pretrain_outputs_mask)
                            model_pretrain_outputs.append(split_pretrain_outputs)
                            split_note_embeddings = self._Attention(
                                    split_outputs, lengths)
                            notes_embeddings.append(split_note_embeddings)
                notes_embeddings = tf.concat(notes_embeddings, axis=0)
                # Gather the bagging IDs for the notes embeddings.
                # The (batch, sequence) index for note i after bagging will be
                #   (notes_orig_indices[i, 0], notes_bagging_ids[i]).
                notes_bagging_ids = tf.to_int32(
                        tf.gather_nd(bagging_ids, notes_orig_indices))
                # Map index pairs to scalar values that can be input to tf.unique().
                max_bag = tf.reduce_max(notes_bagging_ids) + 1
                idx_pair_ids = notes_orig_indices[:, 0] * max_bag + notes_bagging_ids
                # Average the note embeddings that fall in the same bag by finding
                # unique index pairs and using tf.segment_mean.
                unique_pair_ids, unique_pair_indices = tf.unique(idx_pair_ids)
                notes_embeddings = tf.segment_mean(notes_embeddings,
                                                   unique_pair_indices)
                # Invert the mapping to recover the original indices from the unique
                # scalar IDs.
                notes_scatter_indices = tf.stack(
                        [tf.floordiv(unique_pair_ids, max_bag),
                         tf.mod(unique_pair_ids, max_bag)],
                        axis=1)
                batch_seq_shape = tf.stack(
                        [notes_feature.dense_shape[0],
                         tf.reduce_max(processed_seq_length)],
                        axis=0)
                # Scatter the notes embeddings into the correct shape.
                new_dense_shape = tf.to_int32(tf.stack(
                        [batch_seq_shape[0],
                         batch_seq_shape[1],
                         self._notes_model_dim]))
                notes_embeddings_seq = tf.scatter_nd(
                        notes_scatter_indices,
                        notes_embeddings,
                        shape=new_dense_shape)
                # Set embedding size statically for record-level RNN.
                notes_embeddings_seq.set_shape([None, None, self._notes_model_dim])
                seq_outputs.append(notes_embeddings_seq)
                # Merge the split pretrain output tensor.
                # Note that if the notes RNN is bidirectional, the list of pretrain
                # output tensors contains (fw, bw) pairs, and must be flattened to a
                # single list of all fw outputs followed by all bw outputs.
                if self._notes_bidirectional:
                    model_pretrain_outputs = sum(zip(*model_pretrain_outputs), ())
                model_pretrain_outputs = tf.concat(model_pretrain_outputs, axis=0)
            else:
                seq_outputs.append(layer.Forward(processed_seq_features[feature]))

        if processed_dense_features:
            normalized_dense_features = g_ops.normalize_dense(
                    processed_dense_features,
                    dt_unique,
                    self._dense_feature_config_path)
            seq_outputs.append(normalized_dense_features)

        outputs_dict = {'seq_tensor': tf.concat(seq_outputs, 2),
                        'seq_length': processed_seq_length}
        if self._use_notes_model:
            outputs_dict['pretrain_tensor'] = model_pretrain_outputs

        return outputs_dict

