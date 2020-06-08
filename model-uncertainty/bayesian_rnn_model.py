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
from edward2.experimental.rank1_bnns import rank1_bnn_layers
from edward2.experimental.rank1_bnns import utils as rank1_utils
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
               rnn_dim,
               num_rnn_layers,
               hidden_layer_dim,
               output_layer_dim,
               rnn_uncertainty,
               hidden_uncertainty,
               output_uncertainty,
               bias_uncertainty,
               prior_stddev,
               l2,
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
      l2: Amount of L2 regularization to apply to the deterministic weights.
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
            clip_norm,
            rnn_dim,
            kernel_regularizer=make_regularizer(),
            recurrent_regularizer=make_regularizer(),
            bias_initializer=bias_initializer,
            bias_regularizer=make_regularizer(bias_uncertainty))
      else:
        lstm_cell = tf.keras.layers.LSTMCell(
            rnn_dim,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            recurrent_regularizer=tf.keras.regularizers.l2(l2),
            bias_regularizer=tf.keras.regularizers.l2(l2))
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
            self.hidden_layer_dim,
            activation=tf.nn.relu6,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            bias_regularizer=tf.keras.regularizers.l2(l2))

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
          output_layer_dim,
          kernel_regularizer=tf.keras.regularizers.l2(l2),
          bias_regularizer=tf.keras.regularizers.l2(l2))

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
      A Tensor of logits of shape (batch_size, output_layer_dim).
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
               rnn_dim,
               num_rnn_layers,
               hidden_layer_dim,
               output_layer_dim,
               rnn_uncertainty,
               hidden_uncertainty,
               output_uncertainty,
               bias_uncertainty,
               embeddings_uncertainty,
               prior_stddev,
               l2,
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
      l2: Amount of L2 regularization to apply to the deterministic weights.
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
        l2=l2,
        clip_norm=clip_norm,
        return_sequences=return_sequences)

  def call(self, inputs):
    """Runs the model.

    Args:
      inputs: The features to process. A dict of feature names to Tensors or
        SparseTensors. At minimum, must include the features specified in
        self.sequence_features and self.context_features, as well as deltaTime
        and sequenceLength.

    Returns:
      A Tensor of logits of shape (batch_size, output_layer_dim).
    """
    (sequence_embeddings, context_embeddings,
     return_dict) = self.embedding_layer(inputs)
    sequence_length = return_dict[constants.C_SEQUENCE_LENGTH_KEY]
    logits = self.bayesian_rnn(sequence_embeddings, context_embeddings,
                               sequence_length)
    return logits


class Rank1BayesianRNN(tf.keras.Model):
  """Bayesian RNN model with rank-1 distributions."""

  def __init__(self,
               rnn_dim,
               num_rnn_layers,
               hidden_layer_dim,
               output_layer_dim,
               alpha_initializer,
               gamma_initializer,
               alpha_regularizer,
               gamma_regularizer,
               use_additive_perturbation,
               ensemble_size,
               random_sign_init,
               dropout_rate,
               prior_mean,
               prior_stddev,
               l2,
               clip_norm=1.0,
               return_sequences=False):
    """Initializes the model.

    Args:
      rnn_dim: RNN cell output dimensionality.
      num_rnn_layers: Number of stacked RNN cells.
      hidden_layer_dim: Hidden layer output dimensionality.
      output_layer_dim: Output layer output dimensionality.
      alpha_initializer: The initializer for the alpha parameters.
      gamma_initializer: The initializer for the gamma parameters.
      alpha_regularizer: The regularizer for the alpha parameters.
      gamma_regularizer: The regularizer for the gamma parameters.
      use_additive_perturbation: Whether or not to use additive perturbations
        instead of multiplicative perturbations.
      ensemble_size: Number of ensemble members.
      random_sign_init: Value used to initialize trainable deterministic
        initializers, as applicable. Values greater than zero result in
        initialization to a random sign vector, where random_sign_init is the
        probability of a 1 value. Values less than zero result in initialization
        from a Gaussian with mean 1 and standard deviation equal to
        -random_sign_init.
      dropout_rate: Dropout rate.
      prior_mean: Mean of the prior.
      prior_stddev: Standard deviation of the prior.
      l2: Amount of L2 regularization to apply to the deterministic weights.
      clip_norm: Gradient clipping norm value for per-time step clipping within
        the LSTM cell, and for clipping of all aggregated gradients.
      return_sequences: Whether or not to return outputs at each time step from
        the LSTM, rather than just the final time step.
    """
    super().__init__()
    self.hidden_layer_dim = hidden_layer_dim

    # 1. RNN layer.
    cells = []
    for _ in range(num_rnn_layers):
      # TODO(dusenberrymw): Determine if a grad-clipped version is needed.
      lstm_cell = rank1_bnn_layers.LSTMCellRank1(
          rnn_dim,
          alpha_initializer=rank1_utils.make_initializer(
              alpha_initializer, random_sign_init, dropout_rate),
          gamma_initializer=rank1_utils.make_initializer(
              gamma_initializer, random_sign_init, dropout_rate),
          recurrent_alpha_initializer=rank1_utils.make_initializer(
              alpha_initializer, random_sign_init, dropout_rate),
          recurrent_gamma_initializer=rank1_utils.make_initializer(
              gamma_initializer, random_sign_init, dropout_rate),
          alpha_regularizer=rank1_utils.make_regularizer(
              alpha_regularizer, prior_mean, prior_stddev),
          gamma_regularizer=rank1_utils.make_regularizer(
              gamma_regularizer, prior_mean, prior_stddev),
          recurrent_alpha_regularizer=rank1_utils.make_regularizer(
              alpha_regularizer, prior_mean, prior_stddev),
          recurrent_gamma_regularizer=rank1_utils.make_regularizer(
              gamma_regularizer, prior_mean, prior_stddev),
          kernel_regularizer=tf.keras.regularizers.l2(l2),
          recurrent_regularizer=tf.keras.regularizers.l2(l2),
          bias_regularizer=tf.keras.regularizers.l2(l2),
          use_additive_perturbation=use_additive_perturbation,
          ensemble_size=ensemble_size)
      cells.append(lstm_cell)
    self.rnn_layer = tf.keras.layers.RNN(cells, return_sequences=False)

    # 2. Affine layer on combination of RNN output and context features.
    if self.hidden_layer_dim > 0:
      self.hidden_layer = rank1_bnn_layers.DenseRank1(
          self.hidden_layer_dim,
          activation=tf.nn.relu6,
          alpha_initializer=rank1_utils.make_initializer(
              alpha_initializer, random_sign_init, dropout_rate),
          gamma_initializer=rank1_utils.make_initializer(
              gamma_initializer, random_sign_init, dropout_rate),
          kernel_initializer="he_normal",
          alpha_regularizer=rank1_utils.make_regularizer(
              alpha_regularizer, prior_mean, prior_stddev),
          gamma_regularizer=rank1_utils.make_regularizer(
              gamma_regularizer, prior_mean, prior_stddev),
          kernel_regularizer=tf.keras.regularizers.l2(l2),
          bias_regularizer=tf.keras.regularizers.l2(l2),
          use_additive_perturbation=use_additive_perturbation,
          ensemble_size=ensemble_size)

    # 3. Output affine layer.
    self.output_layer = rank1_bnn_layers.DenseRank1(
        output_layer_dim,
        alpha_initializer=rank1_utils.make_initializer(
            alpha_initializer, random_sign_init, dropout_rate),
        gamma_initializer=rank1_utils.make_initializer(
            gamma_initializer, random_sign_init, dropout_rate),
        kernel_initializer="he_normal",
        alpha_regularizer=rank1_utils.make_regularizer(
            alpha_regularizer, prior_mean, prior_stddev),
        gamma_regularizer=rank1_utils.make_regularizer(
            gamma_regularizer, prior_mean, prior_stddev),
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2),
        use_additive_perturbation=use_additive_perturbation,
        ensemble_size=ensemble_size)

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
      A Tensor of logits of shape (batch_size, output_layer_dim).
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


class Rank1BayesianRNNWithEmbeddings(tf.keras.Model):
  """A rank-1 Bayesian RNN model with a Bayesian embedding layer.

  Attributes:
    embedding_layer: A SequenceEmbedding Keras layer.
    bayesian_rnn: A BayesianRNN Keras model.
  """

  def __init__(self,
               embedding_config,
               sequence_features,
               context_features,
               rnn_dim,
               num_rnn_layers,
               hidden_layer_dim,
               output_layer_dim,
               embeddings_initializer,
               embeddings_regularizer,
               alpha_initializer,
               gamma_initializer,
               alpha_regularizer,
               gamma_regularizer,
               use_additive_perturbation,
               ensemble_size,
               random_sign_init,
               dropout_rate,
               prior_mean,
               prior_stddev,
               l2,
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

    All parameters come from `embedding.SequenceEmbedding` and
    `Rank1BayesianRNN`.

    Args:
      embedding_config: An embedding config proto.
      sequence_features: List of the sequential features to process.
      context_features: List of the context (per-sequence) features to process.
      rnn_dim: RNN cell output dimensionality.
      num_rnn_layers: Number of stacked RNN cells.
      hidden_layer_dim: Hidden layer output dimensionality.
      output_layer_dim: Output layer output dimensionality.
      embeddings_initializer: The initializer for the embedding parameters.
      embeddings_regularizer: The regularizer for the embedding parameters.
      alpha_initializer: The initializer for the alpha parameters.
      gamma_initializer: The initializer for the gamma parameters.
      alpha_regularizer: The regularizer for the alpha parameters.
      gamma_regularizer: The regularizer for the gamma parameters.
      use_additive_perturbation: Whether or not to use additive perturbations
        instead of multiplicative perturbations.
      ensemble_size: Number of ensemble members.
      random_sign_init: Value used to initialize trainable deterministic
        initializers, as applicable. Values greater than zero result in
        initialization to a random sign vector, where random_sign_init is the
        probability of a 1 value. Values less than zero result in initialization
        from a Gaussian with mean 1 and standard deviation equal to
        -random_sign_init.
      dropout_rate: Dropout rate.
      prior_mean: Mean of the prior.
      prior_stddev: Standard deviation of the prior.
      l2: Amount of L2 regularization to apply to the deterministic weights.
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
    super().__init__()

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
    self.bayesian_rnn = Rank1BayesianRNN(
        rnn_dim=rnn_dim,
        num_rnn_layers=num_rnn_layers,
        hidden_layer_dim=hidden_layer_dim,
        output_layer_dim=output_layer_dim,
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        alpha_regularizer=alpha_regularizer,
        gamma_regularizer=gamma_regularizer,
        use_additive_perturbation=use_additive_perturbation,
        ensemble_size=ensemble_size,
        random_sign_init=random_sign_init,
        dropout_rate=dropout_rate,
        prior_mean=prior_mean,
        prior_stddev=prior_stddev,
        clip_norm=clip_norm,
        l2=l2,
        return_sequences=return_sequences)

  def call(self, inputs):
    """Runs the model.

    Args:
      inputs: The features to process. A dict of feature names to Tensors or
        SparseTensors. At minimum, must include the features specified in
        self.sequence_features and self.context_features, as well as deltaTime
        and sequenceLength.

    Returns:
      A Tensor of logits of shape (batch_size, output_layer_dim).
    """
    (sequence_embeddings, context_embeddings,
     return_dict) = self.embedding_layer(inputs)
    sequence_length = return_dict[constants.C_SEQUENCE_LENGTH_KEY]
    logits = self.bayesian_rnn(sequence_embeddings, context_embeddings,
                               sequence_length)
    return logits
