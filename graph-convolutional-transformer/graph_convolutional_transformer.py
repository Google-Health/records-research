"""Copyright 2019 Google LLC.

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

import tensorflow as tf
import sys


class FeatureEmbedder(object):
  """This class is used to convert SparseTensor inputs to dense Tensors.

  This class is used to convert raw features to their vector representations.
  It takes in a dictionary, where the key is the name of a feature (e.g.
  diagnosis_id) and the value is a SparseTensor of integers (i.e. lookup IDs),
  then retrieves corresponding vector representations using tf.embedding.lookup.
  """

  def __init__(self, vocab_sizes, feature_keys, embedding_size):
    """Init function.

    Args:
      vocab_sizes: A dictionary of vocabularize sizes for each feature.
      feature_keys: A list of feature names you want to use.
      embedding_size: The dimension size of the feature representation vector.
    """
    self._params = {}
    self._feature_keys = feature_keys
    self._vocab_sizes = vocab_sizes
    dummy_emb = tf.zeros([1, embedding_size], dtype=tf.float32)

    for feature_key in feature_keys:
      vocab_size = self._vocab_sizes[feature_key]
      emb = tf.get_variable(
          feature_key, shape=(vocab_size, embedding_size), dtype=tf.float32)
      self._params[feature_key] = tf.concat([emb, dummy_emb], axis=0)

    self._params['visit'] = tf.get_variable(
        'visit', shape=(1, embedding_size), dtype=tf.float32)

  def lookup(self, feature_map, max_num_codes):
    """Look-up function.

    This function converts the SparseTensor of integers to a dense Tensor of
    tf.float32.

    Args:
      feature_map: A dictionary of SparseTensors for each feature.
      max_num_codes: The maximum number of how many feature there can be inside
        a single visit, per feature. For example, if this is set to 50, then we
        are assuming there can be up to 50 diagnosis codes, 50 treatment codes,
        and 50 lab codes. This will be used for creating the prior matrix.

    Returns:
      embeddings: A dictionary of dense representation Tensors for each feature.
      masks: A dictionary of dense float32 Tensors for each feature, that will
        be used as a mask in the downstream tasks.
    """
    masks = {}
    embeddings = {}
    for key in self._feature_keys:
      if max_num_codes > 0:
        feature = tf.SparseTensor(
            indices=feature_map[key].indices,
            values=feature_map[key].values,
            dense_shape=[
                feature_map[key].dense_shape[0],
                feature_map[key].dense_shape[1], max_num_codes
            ])
      else:
        feature = feature_map[key]
      feature_ids = tf.sparse.to_dense(
          feature, default_value=self._vocab_sizes[key])
      feature_ids = tf.squeeze(feature_ids, axis=1)
      embeddings[key] = tf.nn.embedding_lookup(self._params[key], feature_ids)

      mask = tf.SparseTensor(
          indices=feature.indices,
          values=tf.ones(tf.shape(feature.values)),
          dense_shape=feature.dense_shape)
      masks[key] = tf.squeeze(tf.sparse.to_dense(mask), axis=1)

    batch_size = tf.shape(embeddings.values()[0])[0]
    embeddings['visit'] = tf.tile(self._params['visit'][None, :, :],
                                  [batch_size, 1, 1])

    masks['visit'] = tf.ones(batch_size)[:, None]

    return embeddings, masks


class GraphConvolutionalTransformer(tf.keras.layers.Layer):
  """Graph Convolutional Transformer class.

  This is an implementation of Graph Convolutional Transformer. With a proper
  set of options, it can be used as a vanilla Transformer.
  """

  def __init__(self,
               embedding_size=128,
               num_transformer_stack=3,
               num_feedforward=2,
               num_attention_heads=1,
               ffn_dropout=0.1,
               attention_normalizer='softmax',
               multihead_attention_aggregation='concat',
               directed_attention=False,
               use_inf_mask=True,
               use_prior=True,
               **kwargs):
    """Init function.

    Args:
      embedding_size: The size of the dimension for hidden layers.
      num_transformer_stack: The number of Transformer blocks.
      num_feedforward: The number of layers in the feedforward part of
        Transformer.
      num_attention_heads: The number of attention heads.
      ffn_dropout: Dropout rate used inside the feedforward part.
      attention_normalizer: Use either 'softmax' or 'sigmoid' to normalize the
        attention values.
      multihead_attention_aggregation: Use either 'concat' or 'sum' to handle
        the outputs from multiple attention heads.
      directed_attention: Decide whether you want to use the unidirectional
        attention, where information accumulates inside the dummy visit node.
      use_inf_mask: Decide whether you want to use the guide matrix. Currently
        unused.
      use_prior: Decide whether you want to use the conditional probablility
        information. Currently unused.
      **kwargs: Other arguments to tf.keras.layers.Layer init.
    """

    super(GraphConvolutionalTransformer, self).__init__(**kwargs)
    self._hidden_size = embedding_size
    self._num_stack = num_transformer_stack
    self._num_feedforward = num_feedforward
    self._num_heads = num_attention_heads
    self._ffn_dropout = ffn_dropout
    self._attention_normalizer = attention_normalizer
    self._multihead_aggregation = multihead_attention_aggregation
    self._directed_attention = directed_attention
    self._use_inf_mask = use_inf_mask
    self._use_prior = use_prior

    self._layers = {}
    self._layers['Q'] = []
    self._layers['K'] = []
    self._layers['V'] = []
    self._layers['ffn'] = []
    self._layers['head_agg'] = []

    for i in range(self._num_stack):
      self._layers['Q'].append(
          tf.keras.layers.Dense(
              self._hidden_size * self._num_heads, use_bias=False))
      self._layers['K'].append(
          tf.keras.layers.Dense(
              self._hidden_size * self._num_heads, use_bias=False))
      self._layers['V'].append(
          tf.keras.layers.Dense(
              self._hidden_size * self._num_heads, use_bias=False))

      if self._multihead_aggregation == 'concat':
        self._layers['head_agg'].append(
            tf.keras.layers.Dense(self._hidden_size, use_bias=False))

      self._layers['ffn'].append([])
      # Don't need relu for the last feedforward.
      for _ in range(self._num_feedforward - 1):
        self._layers['ffn'][i].append(
            tf.keras.layers.Dense(self._hidden_size, activation='relu'))
      self._layers['ffn'][i].append(tf.keras.layers.Dense(self._hidden_size))

  def feedforward(self, features, stack_index, training=None):
    """Feedforward component of Transformer.

    Args:
      features: 3D float Tensor of size (batch_size, num_features,
        embedding_size). This is the input embedding to GCT.
      stack_index: An integer to indicate which Transformer block we are in.
      training: Whether to run in training or eval mode.

    Returns:
      Latent representations derived from this feedforward network.
    """
    for i in range(self._num_feedforward):
      features = self._layers['ffn'][stack_index][i](features)
      if training:
        features = tf.nn.dropout(features, rate=self._ffn_dropout)

    return features

  def qk_op(self,
            features,
            stack_index,
            batch_size,
            num_codes,
            attention_mask,
            inf_mask=None,
            directed_mask=None):
    """Attention generation part of Transformer.

    Args:
      features: 3D float Tensor of size (batch_size, num_features,
        embedding_size). This is the input embedding to GCT.
      stack_index: An integer to indicate which Transformer block we are in.
      batch_size: The size of the mini batch.
      num_codes: The number of features (i.e. codes) given as input.
      attention_mask: A Tensor for suppressing the attention on the padded
        tokens.
      inf_mask: The guide matrix to suppress the attention values to zeros for
        certain parts of the attention matrix (e.g. diagnosis codes cannot
        attend to other diagnosis codes).
      directed_mask: If the user wants to only use the upper-triangle of the
        attention for uni-directional attention flow, we use this strictly lower
        triangular matrix filled with infinity.

    Returns:
      The attention distribution derived from the QK operation.
    """

    q = self._layers['Q'][stack_index](features)
    q = tf.reshape(q,
                   [batch_size, num_codes, self._hidden_size, self._num_heads])

    k = self._layers['K'][stack_index](features)
    k = tf.reshape(k,
                   [batch_size, num_codes, self._hidden_size, self._num_heads])

    # Need to transpose q and k to (2, 0, 1)
    q = tf.transpose(q, perm=[0, 3, 1, 2])
    k = tf.transpose(k, perm=[0, 3, 2, 1])
    pre_softmax = tf.matmul(q, k) / tf.sqrt(
        tf.cast(self._hidden_size, tf.float32))

    pre_softmax -= attention_mask[:, None, None, :]

    if inf_mask is not None:
      pre_softmax -= inf_mask[:, None, :, :]

    if directed_mask is not None:
      pre_softmax -= directed_mask

    if self._attention_normalizer == 'softmax':
      attention = tf.nn.softmax(pre_softmax, axis=3)
    else:
      attention = tf.nn.sigmoid(pre_softmax)
    return attention

  def call(self, features, masks, guide=None, prior_guide=None, training=None):
    """This function transforms the input embeddings.

    This function converts the SparseTensor of integers to a dense Tensor of
    tf.float32.

    Args:
      features: 3D float Tensor of size (batch_size, num_features,
        embedding_size). This is the input embedding to GCT.
      masks: 3D float Tensor of size (batch_size, num_features, 1). This holds
        binary values to indicate which parts are padded and which are not.
      guide: 3D float Tensor of size (batch_size, num_features, num_features).
        This is the guide matrix.
      prior_guide: 3D float Tensor of size (batch_size, num_features,
        num_features). This is the conditional probability matrix.
      training: Whether to run in training or eval mode.

    Returns:
      features: The final layer of GCT.
      attentions: List of attention values from all layers of GCT. This will be
        used later to regularize the self-attention process.
    """

    batch_size = tf.shape(features)[0]
    num_codes = tf.shape(features)[1]

    # Use the given masks to create a negative infinity Tensor to suppress the
    # attention weights of the padded tokens. Note that the given masks has
    # the shape (batch_size, num_codes, 1), so we remove the last dimension
    # during the process.
    mask_idx = tf.cast(tf.where(tf.equal(masks[:, :, 0], 0.)), tf.int32)
    mask_matrix = tf.fill([tf.shape(mask_idx)[0]], tf.float32.max)
    attention_mask = tf.scatter_nd(
        indices=mask_idx, updates=mask_matrix, shape=tf.shape(masks[:, :, 0]))

    inf_mask = None
    if self._use_inf_mask:
      guide_idx = tf.cast(tf.where(tf.equal(guide, 0.)), tf.int32)
      inf_matrix = tf.fill([tf.shape(guide_idx)[0]], tf.float32.max)
      inf_mask = tf.scatter_nd(
          indices=guide_idx, updates=inf_matrix, shape=tf.shape(guide))

    directed_mask = None
    if self._directed_attention:
      inf_matrix = tf.fill([num_codes, num_codes], tf.float32.max)
      inf_matrix = tf.matrix_set_diag(inf_matrix, tf.zeros(num_codes))
      directed_mask = tf.matrix_band_part(inf_matrix, -1, 0)[None, None, :, :]

    attention = None
    attentions = []
    for i in range(self._num_stack):
      features = masks * features

      if self._use_prior and i == 0:
        attention = tf.tile(prior_guide[:, None, :, :],
                            [1, self._num_heads, 1, 1])
      else:
        attention = self.qk_op(features, i, batch_size, num_codes,
                               attention_mask, inf_mask, directed_mask)

      attentions.append(attention)

      v = self._layers['V'][i](features)
      v = tf.reshape(
          v, [batch_size, num_codes, self._hidden_size, self._num_heads])
      v = tf.transpose(v, perm=[0, 3, 1, 2])
      # post_attention is (batch, num_heads, num_codes, hidden_size)
      post_attention = tf.matmul(attention, v)

      if self._num_heads == 1:
        post_attention = tf.squeeze(post_attention, axis=1)
      elif self._multihead_aggregation == 'concat':
        # post_attention is (batch, num_codes, num_heads, hidden_size)
        post_attention = tf.transpose(post_attention, perm=[0, 2, 1, 3])
        # post_attention is (batch, num_codes, num_heads*hidden_size)
        post_attention = tf.reshape(post_attention, [batch_size, num_codes, -1])
        # post attention is (batch, num_codes, hidden_size)
        post_attention = self._layers['head_agg'][i](post_attention)
      else:
        post_attention = tf.reduce_sum(post_attention, axis=1)

      # Residual connection + layer normalization
      post_attention += features
      post_attention = tf.contrib.layers.layer_norm(
          post_attention, begin_norm_axis=2)

      # Feedforward component + residual connection + layer normalization
      post_ffn = self.feedforward(post_attention, i, training)
      post_ffn += post_attention
      post_ffn = tf.contrib.layers.layer_norm(post_ffn, begin_norm_axis=2)

      features = post_ffn

    return features * masks, attentions


def create_matrix_vdpl(features, mask, use_prior, use_inf_mask, max_num_codes,
                       prior_scalar):
  """Creates guide matrix and prior matrix when feature_set='vdpl'.

  This function creates the guide matrix and the prior matrix when visits
  include diagnosis codes, treatment codes, and lab codes.

  Args:
    features: A dictionary of SparseTensors for each feature.
    mask: 3D float Tensor of size (batch_size, num_features, 1). This holds
      binary values to indicate which parts are padded and which are not.
    use_prior: Whether to create the prior matrix.
    use_inf_mask : Whether to create the guide matrix.
    max_num_codes: The maximum number of how many feature there can be inside a
      single visit, per feature. For example, if this is set to 50, then we are
      assuming there can be up to 50 diagnosis codes, 50 treatment codes, and 50
      lab codes. This will be used for creating the prior matrix.
    prior_scalar: A float value between 0.0 and 1.0 to be used to hard-code the
      diagnoal elements of the prior matrix.

  Returns:
    guide: The guide matrix.
    prior_guide: The conditional probablity matrix.
  """
  dx_ids = features['dx_ints']
  proc_ids = features['proc_ints']
  lab_ids = features['loinc_bucketized_ints']

  batch_size = dx_ids.dense_shape[0]
  num_dx_ids = max_num_codes if use_prior else dx_ids.dense_shape[-1]
  num_proc_ids = max_num_codes if use_prior else proc_ids.dense_shape[-1]
  num_lab_ids = max_num_codes if use_prior else lab_ids.dense_shape[-1]
  num_codes = 1 + num_dx_ids + num_proc_ids + num_lab_ids

  guide = None
  if use_inf_mask:
    row0 = tf.concat([
        tf.zeros([1, 1]),
        tf.ones([1, num_dx_ids]),
        tf.zeros([1, num_proc_ids + num_lab_ids])
    ],
                     axis=1)

    row1 = tf.concat([
        tf.zeros([num_dx_ids, 1 + num_dx_ids]),
        tf.ones([num_dx_ids, num_proc_ids]),
        tf.zeros([num_dx_ids, num_lab_ids])
    ],
                     axis=1)

    row2 = tf.concat([
        tf.zeros([num_proc_ids, 1 + num_dx_ids + num_proc_ids]),
        tf.ones([num_proc_ids, num_lab_ids])
    ],
                     axis=1)

    row3 = tf.zeros([num_lab_ids, num_codes])

    guide = tf.concat([row0, row1, row2, row3], axis=0)
    guide = guide + tf.transpose(guide)
    guide = tf.tile(guide[None, :, :], [batch_size, 1, 1])
    guide = (
        guide * mask[:, :, None] * mask[:, None, :] +
        tf.eye(num_codes)[None, :, :])

  prior_guide = None
  if use_prior:
    prior_values = features['prior_values']
    prior_idx_values = prior_values.values

    prior_indices = features['prior_indices']
    prior_batch_idx = prior_indices.indices[:, 0][::2]
    prior_idx = tf.reshape(prior_indices.values, [-1, 2])
    prior_idx = tf.concat(
        [prior_batch_idx[:, None], prior_idx[:, :1], prior_idx[:, 1:]], axis=1)

    temp_idx = (
        prior_idx[:, 0] * 1000000 + prior_idx[:, 1] * 1000 + prior_idx[:, 2])
    sorted_idx = tf.contrib.framework.argsort(temp_idx)
    prior_idx = tf.gather(prior_idx, sorted_idx)

    prior_idx_shape = [batch_size, max_num_codes * 3, max_num_codes * 3]
    sparse_prior = tf.SparseTensor(
        indices=prior_idx, values=prior_idx_values, dense_shape=prior_idx_shape)
    prior_guide = tf.sparse.to_dense(sparse_prior, validate_indices=True)

    visit_guide = tf.convert_to_tensor(
        [prior_scalar] * max_num_codes + [0.0] * max_num_codes * 2,
        dtype=tf.float32)
    prior_guide = tf.concat(
        [tf.tile(visit_guide[None, None, :], [batch_size, 1, 1]), prior_guide],
        axis=1)
    visit_guide = tf.concat([[0.0], visit_guide], axis=0)
    prior_guide = tf.concat(
        [tf.tile(visit_guide[None, :, None], [batch_size, 1, 1]), prior_guide],
        axis=2)
    prior_guide = (
        prior_guide * mask[:, :, None] * mask[:, None, :] +
        prior_scalar * tf.eye(num_codes)[None, :, :])
    degrees = tf.reduce_sum(prior_guide, axis=2)
    prior_guide = prior_guide / degrees[:, :, None]

  return guide, prior_guide


def create_matrix_vdp(features, mask, use_prior, use_inf_mask, max_num_codes,
                      prior_scalar):
  """Creates guide matrix and prior matrix when feature_set='vdp'.

  This function creates the guide matrix and the prior matrix when visits
  include diagnosis codes, treatment codes, but not lab codes.

  Args:
    features: A dictionary of SparseTensors for each feature.
    mask: 3D float Tensor of size (batch_size, num_features, 1). This holds
      binary values to indicate which parts are padded and which are not.
    use_prior: Whether to create the prior matrix.
    use_inf_mask : Whether to create the guide matrix.
    max_num_codes: The maximum number of how many feature there can be inside a
      single visit, per feature. For example, if this is set to 50, then we are
      assuming there can be up to 50 diagnosis codes and 50 treatment codes.
      This will be used for creating the prior matrix.
    prior_scalar: A float value between 0.0 and 1.0 to be used to hard-code the
      diagnoal elements of the prior matrix.

  Returns:
    guide: The guide matrix.
    prior_guide: The conditional probablity matrix.
  """
  dx_ids = features['dx_ints']
  proc_ids = features['proc_ints']

  batch_size = dx_ids.dense_shape[0]
  num_dx_ids = max_num_codes if use_prior else dx_ids.dense_shape[-1]
  num_proc_ids = max_num_codes if use_prior else proc_ids.dense_shape[-1]
  num_codes = 1 + num_dx_ids + num_proc_ids

  guide = None
  if use_inf_mask:
    row0 = tf.concat([
        tf.zeros([1, 1]),
        tf.ones([1, num_dx_ids]),
        tf.zeros([1, num_proc_ids])
    ],
                     axis=1)

    row1 = tf.concat([
        tf.zeros([num_dx_ids, 1 + num_dx_ids]),
        tf.ones([num_dx_ids, num_proc_ids])
    ],
                     axis=1)

    row2 = tf.zeros([num_proc_ids, num_codes])

    guide = tf.concat([row0, row1, row2], axis=0)
    guide = guide + tf.transpose(guide)
    guide = tf.tile(guide[None, :, :], [batch_size, 1, 1])
    guide = (
        guide * mask[:, :, None] * mask[:, None, :] +
        tf.eye(num_codes)[None, :, :])

  prior_guide = None
  if use_prior:
    prior_values = features['prior_values']
    prior_idx_values = prior_values.values

    prior_indices = features['prior_indices']
    prior_batch_idx = prior_indices.indices[:, 0][::2]
    prior_idx = tf.reshape(prior_indices.values, [-1, 2])
    prior_idx = tf.concat(
        [prior_batch_idx[:, None], prior_idx[:, :1], prior_idx[:, 1:]], axis=1)

    temp_idx = (
        prior_idx[:, 0] * 1000000 + prior_idx[:, 1] * 1000 + prior_idx[:, 2])
    sorted_idx = tf.contrib.framework.argsort(temp_idx)
    prior_idx = tf.gather(prior_idx, sorted_idx)

    prior_idx_shape = [batch_size, max_num_codes * 2, max_num_codes * 2]
    sparse_prior = tf.SparseTensor(
        indices=prior_idx, values=prior_idx_values, dense_shape=prior_idx_shape)
    prior_guide = tf.sparse.to_dense(sparse_prior, validate_indices=True)

    visit_guide = tf.convert_to_tensor(
        [prior_scalar] * max_num_codes + [0.0] * max_num_codes * 1,
        dtype=tf.float32)
    prior_guide = tf.concat(
        [tf.tile(visit_guide[None, None, :], [batch_size, 1, 1]), prior_guide],
        axis=1)
    visit_guide = tf.concat([[0.0], visit_guide], axis=0)
    prior_guide = tf.concat(
        [tf.tile(visit_guide[None, :, None], [batch_size, 1, 1]), prior_guide],
        axis=2)
    prior_guide = (
        prior_guide * mask[:, :, None] * mask[:, None, :] +
        prior_scalar * tf.eye(num_codes)[None, :, :])
    degrees = tf.reduce_sum(prior_guide, axis=2)
    prior_guide = prior_guide / degrees[:, :, None]

  return guide, prior_guide


class SequenceExampleParser(object):
  """A very simple SequenceExample parser for eICU data.

  This Parser class is intended to be used for eICU SequenceExamples obtained
  from process_eicu.py. This class will not work with synthetic samples obtained
  from process_synthetic.py, because synthetic samples contain a different set
  of features and labels than eICU samples.
  """

  def __init__(self, batch_size, num_map_threads=4):
    """Init function."""
    self.context_features_config = {
        'patientId': tf.VarLenFeature(tf.string),
        'label.readmission': tf.FixedLenFeature([1], tf.int64),
        'label.expired': tf.FixedLenFeature([1], tf.int64)
    }

    self.sequence_features_config = {
        'dx_ints': tf.VarLenFeature(tf.int64),
        'proc_ints': tf.VarLenFeature(tf.int64),
        'prior_indices': tf.VarLenFeature(tf.int64),
        'prior_values': tf.VarLenFeature(tf.float32)
    }

    self.batch_size = batch_size
    self.num_map_threads = num_map_threads

  def __call__(self, tfrecord_path, label_key, training):
    """Parse function.

    Args:
      tfrecord_path: Path to TFRecord of SequenceExamples.
      training: Boolean value to indicate whether the model if training.

    Returns:
      Dataset iterator.
    """

    def parser_fn(serialized_example):
      (batch_context, batch_sequence) = tf.io.parse_single_sequence_example(
          serialized_example,
          context_features=self.context_features_config,
          sequence_features=self.sequence_features_config)
      labels = tf.squeeze(tf.cast(batch_context[label_key], tf.float32))
      return batch_sequence, labels

    num_epochs = None if training else 1
    buffer_size = self.batch_size * 32
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parser_fn, num_parallel_calls=self.num_map_threads)
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(1)

    return dataset.make_one_shot_iterator().get_next()


class EHRTransformer(object):
  """Transformer-based EHR encounter modeling algorithm.

  All features within each encounter are put through multiple steps of
  self-attention. There is a dummy visit embedding in addition to other
  feature embeddings, which can be used for encounter-level predictions.
  """

  def __init__(self,
               gct_params,
               feature_keys=['dx_ints', 'proc_ints'],
               label_key='label.readmission',
               vocab_sizes={'dx_ints':3249, 'proc_ints':2210},
               feature_set='vdp',
               max_num_codes=50,
               prior_scalar=0.5,
               reg_coef=0.1,
               num_classes=1,
               learning_rate=1e-3,
               batch_size=32):
    """Init function.

    Args:
      gct_params: A dictionary parameteres to be used inside GCT class. See GCT
        comments for more information.
      feature_keys: A list of feature names you want to use. (e.g. ['dx_ints,
        'proc_ints', 'lab_ints'])
      vocab_sizes: A dictionary of vocabularize sizes for each feature. (e.g.
        {'dx_ints': 1001, 'proc_ints': 1001, 'lab_ints': 1001})
      feature_set: Use 'vdpl' to indicate your features are diagnosis codes,
        treatment codes, and lab codes. Use 'vdp' to indicate your features are
        diagnosis codes and treatment codes.
      max_num_codes: The maximum number of how many feature there can be inside
        a single visit, per feature. For example, if this is set to 50, then we
        are assuming there can be up to 50 diagnosis codes, 50 treatment codes,
        and 50 lab codes. This will be used for creating the prior matrix.
      prior_scalar: A float value between 0.0 and 1.0 to be used to hard-code
        the diagnoal elements of the prior matrix.
      reg_coef: A coefficient to decide the KL regularization balance when
        training GCT.
      num_classes: This is set to 1, because this implementation only supports
        graph-level binary classification.
      learning_rate: Learning rate for Adam optimizer.
      batch_size: Batch size.
    """
    self._feature_keys = feature_keys
    self._label_key = label_key
    self._vocab_sizes = vocab_sizes
    self._feature_set = feature_set
    self._max_num_codes = max_num_codes
    self._prior_scalar = prior_scalar
    self._reg_coef = reg_coef
    self._num_classes = num_classes
    self._learning_rate = learning_rate
    self._batch_size = batch_size

    self._gct_params = gct_params
    self._embedding_size = gct_params['embedding_size']
    self._num_transformer_stack = gct_params['num_transformer_stack']
    self._use_inf_mask = gct_params['use_inf_mask']
    self._use_prior = gct_params['use_prior']

    self._seqex_reader = SequenceExampleParser(self._batch_size)

  def get_prediction(self, model, feature_embedder, features, training=False):
    """Accepts features and produces logits and attention values.

    Args:
      features: A dictionary of SparseTensors for each sequence feature.
      training: A boolean value to indicate whether the predictions are for
        training or inference. If set to True, dropouts will take effect.

    Returns:
      logits: Logits for prediction.
      attentions: List of attention values from all layers of GCT. Pass this to
        get_loss to regularize the attention generation mechanism.
    """
    # 1. Embedding lookup
    embedding_dict, mask_dict = feature_embedder.lookup(
        features, self._max_num_codes)

    # 2. Concatenate embeddings and masks into a single tensor.
    keys = ['visit'] + self._feature_keys
    embeddings = tf.concat([embedding_dict[key] for key in keys], axis=1)
    masks = tf.concat([mask_dict[key] for key in keys], axis=1)

    # 2-1. Create the guide matrix and the prior matrix.
    if self._feature_set == 'vdpl':
      guide, prior_guide = create_matrix_vdpl(features, masks, self._use_prior,
                                              self._use_inf_mask,
                                              self._max_num_codes,
                                              self._prior_scalar)
    elif self._feature_set == 'vdp':
      guide, prior_guide = create_matrix_vdp(features, masks, self._use_prior,
                                             self._use_inf_mask,
                                             self._max_num_codes,
                                             self._prior_scalar)
    else:
      sys.exit(0)

    # 3. Process embeddings with GCT
    hidden, attentions = model(
        embeddings, masks[:, :, None], guide, prior_guide, training)

    # 4. Generate logits
    pre_logit = hidden[:, 0, :]
    pre_logit = tf.reshape(pre_logit, [-1, self._embedding_size])
    logits = tf.layers.dense(pre_logit, self._num_classes, activation=None)
    logits = tf.squeeze(logits)

    return logits, attentions

  def get_loss(self, logits, labels, attentions):
    """Creates a loss tensor.

    Args:
      logits: Logits for prediction. This is obtained by calling get_prediction.
      labels: Labels for prediction.
      attentions: List of attention values from all layers of GCT. This is
        obtained by calling get_prediction.

    Returns:
      Loss tensor. If we use the conditional probability matrix, then GCT's
      attention mechanism will be regularized using KL divergence.
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)

    if self._use_prior:
      kl_terms = []
      attention_tensor = tf.convert_to_tensor(attentions)
      for i in range(1, self._num_transformer_stack):
        log_p = tf.log(attention_tensor[i - 1] + 1e-12)
        log_q = tf.log(attention_tensor[i] + 1e-12)
        kl_term = attention_tensor[i - 1] * (log_p - log_q)
        kl_term = tf.reduce_sum(kl_term, axis=-1)
        kl_term = tf.reduce_mean(kl_term)
        kl_terms.append(kl_term)

      reg_term = tf.reduce_mean(kl_terms)
      loss += self._reg_coef * reg_term

    return loss

  def input_fn(self, tfrecord_path, training):
    """Input function to be used by TensorFlow Estimator.

    Args:
      tfrecord_path: Path to TFRecord of SequenceExamples.
      training: Boolean value to indicate whether the model if training.

    Return:
      Input generator.
    """
    return self._seqex_reader(tfrecord_path, self._label_key, training)

  def model_fn(self, features, labels, mode):
    """Model function to be used by TensorFlow Estimator.

    Args:
      features: Dictionary of features.
      labels: True labels for training.
      mode: The mode the model is in tf.estimator.ModeKeys.

    Return:
      Train/Eval/Prediction op depending on the mode.
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    model = GraphConvolutionalTransformer(**self._gct_params)
    feature_embedder = FeatureEmbedder(
        self._vocab_sizes, self._feature_keys, self._embedding_size)

    logits, attentions = self.get_prediction(
        model, feature_embedder, features, training)
    probs = tf.nn.sigmoid(logits)
    predictions = {
        'probabilities': probs,
        'logits': logits,
    }

    # output predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # create loss (should be equal to caffe softmaxwithloss)
    loss = self.get_loss(logits, labels, attentions)

    if training:
      optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate,
                                         beta1=0.9, beta2=0.999, epsilon=1e-8)
      train_op = optimizer.minimize(loss)
      global_step = tf.train.get_global_step()
      update_global_step = tf.assign(
          global_step, global_step+1, name='update_global_step')

      # create estimator training spec.
      return tf.estimator.EstimatorSpec(
          mode,
          loss=loss,
          train_op=tf.group(train_op, update_global_step),
          predictions=predictions)

    if mode == tf.estimator.ModeKeys.EVAL:
      # Define the metrics:
      metrics_dict = {
          'AUC-PR': tf.metrics.auc(labels, probs, curve='PR',
                                   summation_method='careful_interpolation'),
          'AUC-ROC': tf.metrics.auc(labels, probs, curve='ROC',
                                 summation_method='careful_interpolation')
      }

      #return eval spec
      return tf.estimator.EstimatorSpec(
          mode, loss=loss, eval_metric_ops=metrics_dict)
