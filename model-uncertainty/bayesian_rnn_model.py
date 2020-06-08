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

"""Bayesian RNN model implementation.

For the binary mortality problem, the sequential model parameterizes a Bernoulli
distribution over the binary outcome, where the Bernoulli for a given prediction
represents data uncertainty for that example.  By introducing weight
uncertainty, we can induce a distribution over the parameter of this Bernoulli
distribution, allowing us to represent model uncertainty and determine credible
intervals for any given prediction.

On all parameterized layers, we (optionally) place an independent standard
Normal distribution over each value in the weight vectors, i.e., a joint prior
distribution of independent Gaussians.  We can then aim to compute a posterior
distribution for these weights given observed data.  For tractability, we make
use of a variational approach and introduce an approximate posterior `q(w |
theta)` that has the same form as the prior.  Each Gaussian distribution in the
posterior is initialized to a Normal distribution centered at approximately zero
with a small standard deviation (usually of the same form as in a He Normal
initialization.  We then optimize a variational lower bound term, `KL[q(w |
theta) || p(w | y, x)] = KL[q(w | theta) || p(w)] - log p(y | w, x)`, with
respect to `theta` and any other deterministic weights, collectively known as
the parameters.  At prediction time, we can sample predictions from the model
for a given example to empirically determine the distribution over the
parameterization of the predictive distribution.
"""
import edward2 as ed
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

# TODO(dusenberrymw): Expose open-source versions of these files.
import constants
import util
import embedding


@tf.custom_gradient
def _clip_gradient(x, clip_norm):
  """Identity function that performs gradient clipping."""
  def grad(dy):
    # NOTE: Must return a gradient for all inputs to `clip_gradient`.
    return tf.clip_by_global_norm([dy], clip_norm)[0][0], tf.constant(0.)
  return x, grad


class LSTMCellReparameterizationGradClipped(
    ed.layers.LSTMCellReparameterization):
  """Bayesian LSTMCell with per-time step gradient clipping."""

  def __init__(self, clip_norm, *args, **kwargs):
    self.clip_norm = clip_norm
    super(LSTMCellReparameterizationGradClipped, self).__init__(*args, **kwargs)

  def call(self, inputs, states, training=None):
    with tf.name_scope("clip_gradient"):
      inputs = _clip_gradient(inputs, self.clip_norm)
      states = [_clip_gradient(x, self.clip_norm) for x in states]
    return super(LSTMCellReparameterizationGradClipped, self).call(
        inputs, states, training)


class BayesianRNN(tf.keras.Model):
  """Bayesian RNN model."""

  def __init__(self,
               rnn_dim=512,
               num_rnn_layers=1,
               hidden_layer_dim=128,
               output_layer_dim=1,
               rnn_uncertainty=True,
               hidden_uncertainty=True,
               output_uncertainty=True,
               bias_uncertainty=False,
               prior_stddev=1.,
               clip_norm=1.0,
               return_sequences=False):
    """Initializes the model.

    Args:
      rnn_dim: RNN cell output dimensionality.
      num_rnn_layers: Number of stacked RNN cells.
      hidden_layer_dim: Hidden layer output dimensionality.
      output_layer_dim: Output layer output dimensionality.
      rnn_uncertainty: Whether or not to use a Bayesian RNN layer.
      hidden_uncertainty: Whether or not to use a Bayesian hidden dense layer.
      output_uncertainty: Whether or not to use a Bayesian output dense layer.
      bias_uncertainty: Whether or not to use Bayesian bias terms in all of the
        Bayesian layers.
      prior_stddev: The standard deviation for the Normal prior.
      clip_norm: Gradient clipping norm value for per-time step clipping within
        the LSTM cell, and for clipping of all aggregated gradients.
      return_sequences: Whether or not to return outputs at each time step from
        the LSTM, rather than just the final time step.
    """
    super(BayesianRNN, self).__init__()
    self.hidden_layer_dim = hidden_layer_dim

    def make_regularizer(uncertainty=True):
      if uncertainty:
        return ed.regularizers.normal_kl_divergence(stddev=prior_stddev)
      else:
        return None

    bias_initializer = "trainable_normal" if bias_uncertainty else "zeros"

    # 1. RNN layer.
    cells = []
    for _ in range(num_rnn_layers):
      if rnn_uncertainty:
        lstm_cell = LSTMCellReparameterizationGradClipped(
            clip_norm, rnn_dim,
            kernel_regularizer=make_regularizer(),
            bias_initializer=bias_initializer,
            bias_regularizer=make_regularizer(bias_uncertainty))
      else:
        lstm_cell = tf.keras.layers.LSTMCell(
            rnn_dim, kernel_regularizer=tf.keras.regularizers.l2(l=1.0),
            recurrent_regularizer=tf.keras.regularizers.l2(l=1.0))
      cells.append(lstm_cell)
    self.rnn_layer = tf.keras.layers.RNN(cells, return_sequences=False)

    # 2. Affine layer on combination of RNN output and context features.
    if self.hidden_layer_dim > 0:
      if hidden_uncertainty:
        self.hidden_layer = ed.layers.DenseReparameterization(
            self.hidden_layer_dim,
            activation=tf.nn.relu6,
            kernel_initializer="trainable_he_normal",
            kernel_regularizer=make_regularizer(),
            bias_initializer=bias_initializer,
            bias_regularizer=make_regularizer(bias_uncertainty))
      else:
        self.hidden_layer = tf.keras.layers.Dense(
            self.hidden_layer_dim, activation=tf.nn.relu6,
            kernel_regularizer=tf.keras.regularizers.l2(l=1.0))

    # 3. Output layer.
    self.output_uncertainty = output_uncertainty
    if self.output_uncertainty:
      self.output_layer = ed.layers.DenseReparameterization(
          output_layer_dim,
          kernel_regularizer=make_regularizer(),
          bias_initializer=bias_initializer,
          bias_regularizer=make_regularizer(bias_uncertainty))
    else:
      self.output_layer = tf.keras.layers.Dense(
          output_layer_dim, kernel_regularizer=tf.keras.regularizers.l2(l=1.0))

  def call(self, sequence_embeddings, context_embeddings, sequence_length):
    """Runs the model.

    Args:
      sequence_embeddings: Tensor of shape [batch_size, bagged_seq_length,
        sum(embed_dim)]. The embeddings for all categorical sequence features,
        concatenated along the embedding axis. If dense features are used, the
        embeddings or normalized values are also concatenated.
      context_embeddings: Tensor of shape [batch_size, sum(embed_dim)]. The
        concatenated embeddings for all context features. Includes the patient's
        age if birthdate and prediction time are available in inputs.
      sequence_length: Tensor of shape [batch]. The lengths of the sequences
        after bagging.

    Returns:
      A tuple of (logits Tensor of shape (batch_size, output_layer_dim),
      logits tfp.distribution).
    """
    mask = tf.sequence_mask(sequence_length,
                            maxlen=tf.shape(sequence_embeddings)[1],
                            dtype=sequence_embeddings.dtype)
    mask = tf.reshape(mask, tf.concat(
        [tf.shape(sequence_embeddings)[:-1], [1]], axis=-1))
    last_output = self.rnn_layer(sequence_embeddings, mask=mask)

    if context_embeddings is not None:
      combined_features = tf.concat([last_output, context_embeddings], axis=-1)
    else:
      combined_features = last_output

    if self.hidden_layer_dim > 0:
      combined_features = self.hidden_layer(combined_features)

    return self.output_layer(combined_features)  # shape (n, d)


class BayesianRNNWithEmbeddings(tf.keras.Model):
  """A Bayesian RNN model with a Bayesian embedding layer.

  Attributes:
    embedding_layer: A SequenceEmbedding Keras layer.
    bayesian_rnn: A BayesianRNN Keras model.
  """

  def __init__(self,
               embedding_config,
               sequence_features,
               context_features,
               rnn_dim=512,
               num_rnn_layers=1,
               hidden_layer_dim=128,
               output_layer_dim=1,
               rnn_uncertainty=True,
               hidden_uncertainty=True,
               output_uncertainty=True,
               bias_uncertainty=False,
               embeddings_uncertainty=True,
               prior_stddev=1.,
               clip_norm=1.0,
               return_sequences=False,
               bagging_time_precision=86400,
               bagging_aggregate_older_than=-1,
               embedding_dimension_multiplier=1.0,
               dense_feature_name=None,
               dense_feature_value=None,
               dense_feature_unit=None,
               dense_embedding_dimension=128,
               top_n_dense=10000,
               num_ids_per_dense_feature=5,
               dense_stats_config_path=""):
    """Initializes the model.

    Aside from `embeddings_uncertainty`, all parameters come from
    `embedding.SequenceEmbedding` and `BayesianRNN`.

    Args:
      embedding_config: An embedding config proto.
      sequence_features: List of the sequential features to process.
      context_features: List of the context (per-sequence) features to process.
      rnn_dim: RNN cell output dimensionality.
      num_rnn_layers: Number of stacked RNN cells.
      hidden_layer_dim: Hidden layer output dimensionality.
      output_layer_dim: Output layer output dimensionality.
      rnn_uncertainty: Whether or not to use a Bayesian RNN layer.
      hidden_uncertainty: Whether or not to use a Bayesian hidden dense layer.
      output_uncertainty: Whether or not to use a Bayesian output dense layer.
      bias_uncertainty: Whether or not to use Bayesian bias terms in all of the
        Bayesian layers.
      embeddings_uncertainty: Whether or not to use a Bayesian embedding layer.
      prior_stddev: The standard deviation for the Normal prior.
      clip_norm: Gradient clipping norm value for per-time step clipping within
        the LSTM cell, and for clipping of all aggregated gradients.
      return_sequences: Whether or not to return outputs at each time step from
        the LSTM, rather than just the final time step.
      bagging_time_precision: Precision for bagging in seconds. For example,
        86400 represents day-level bagging.
      bagging_aggregate_older_than: Events older than this (in seconds) will all
        be aggregated into the same bag. `-1` indicates no aggregation.
      embedding_dimension_multiplier: Multiplier for the default embedding
        dimension specified in embedding_config. Applied to all categorical
        features.
      dense_feature_name: A feature in sequence_features representing the names
        of all continuous inputs. Used for dense embedding or normalization.
      dense_feature_value: A feature in sequence_features representing the
        values of all continuous inputs. Used for dense embedding or
        normalization.
      dense_feature_unit: A feature in sequence_features representing the units
        of all continuous input values. Used for dense embedding or
        normalization.
      dense_embedding_dimension: If embed_dense, size of embeddings to use.
      top_n_dense: Number of top dense features (by frequency) to retain for
        model input.
      num_ids_per_dense_feature: If embed_dense, number of embedding IDs per
        feature.
      dense_stats_config_path: Path to stats config for dense features.
    """
    super(BayesianRNNWithEmbeddings, self).__init__()

    if embeddings_uncertainty:
      embeddings_initializer = "trainable_normal"
      embeddings_regularizer = "normal_kl_divergence"
    else:
      embeddings_initializer = "uniform"
      embeddings_regularizer = None

    self.embedding_layer = embedding.SequenceEmbedding(
        embedding_config=embedding_config,
        sequence_features=sequence_features,
        context_features=context_features,
        bagging_time_precision=bagging_time_precision,
        bagging_aggregate_older_than=bagging_aggregate_older_than,
        embedding_dimension_multiplier=embedding_dimension_multiplier,
        dense_feature_name=dense_feature_name,
        dense_feature_value=dense_feature_value,
        dense_feature_unit=dense_feature_unit,
        dense_embedding_dimension=dense_embedding_dimension,
        top_n_dense=top_n_dense,
        num_ids_per_dense_feature=num_ids_per_dense_feature,
        dense_stats_config_path=dense_stats_config_path,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)
    self.bayesian_rnn = BayesianRNN(
        rnn_dim=rnn_dim,
        num_rnn_layers=num_rnn_layers,
        hidden_layer_dim=hidden_layer_dim,
        output_layer_dim=output_layer_dim,
        rnn_uncertainty=rnn_uncertainty,
        hidden_uncertainty=hidden_uncertainty,
        output_uncertainty=output_uncertainty,
        bias_uncertainty=bias_uncertainty,
        prior_stddev=prior_stddev,
        clip_norm=clip_norm,
        return_sequences=return_sequences)

  def call(self, inputs):
    """Runs the model."""
    # TODO(dusenberrymw): Currently, attempting to execute the embedding layer
    # on a GPU results in an error due to not being able to copy a Tensor with
    # type string to the GPU.
    with tf.device("/cpu:0"):
      (sequence_embeddings, context_embeddings,
       return_dict) = self.embedding_layer(inputs)
      sequence_length = return_dict[constants.C_SEQUENCE_LENGTH_KEY]
    logits = self.bayesian_rnn(sequence_embeddings, context_embeddings,
                               sequence_length)
    return logits
