# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal Transformer sequence model implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import hparam
import tensorflow.compat.v1 as tf


# If you only care about the learned multimodal model architecture, this is the
# function you should focus on.
def mufasa_model(input_tensors, hparams):
  """MUFASA model architecture code.

  Args:
    input_tensors: A list of input tensors. The first tensor is categorical
    features, the second tensor is ocntunous features and the third tensor is
    clinical notes. All tensors in shape of [batch_size, seq_len, hidden_dim].
    hparams: A tf.HParams object with model hyperparameters.

  Returns:
    A model processed output with shape [batch_size, seq_len, hidden_dim].
  """
  categorical_data = tf.layers.dense(input_tensors[0], 128)
  continuous_features = tf.layers.dense(input_tensors[1], 128)
  clinical_notes = tf.layers.dense(input_tensors[2], 128)
  dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, 'attention_dropout_broadcast_dims', '')))
  hparams.norm_type = 'layer'

  # Continuous Features Branch.
  continuous_res = continuous_features
  # Layer Norm.
  continuous_hs = common_layers.layer_preprocess(continuous_features, hparams)
  # Self Attention.
  with tf.variable_scope('continuous_attention'):
    continuous_hs = common_attention.multihead_attention(
        query_antecedent=continuous_hs,
        memory_antecedent=None,
        bias=None,
        total_key_depth=128,
        total_value_depth=128,
        output_depth=128,
        num_heads=8,
        dropout_rate=hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        max_relative_position=hparams.max_relative_position,
        dropout_broadcast_dims=dropout_broadcast_dims)
  # Leaky Relu.
  continuous_hs = tf.nn.leaky_relu(continuous_hs)
  # Residul.
  continuous_hs += continuous_res
  continuous_res = continuous_hs
  # Layer Norm.
  continuous_hs = common_layers.layer_preprocess(continuous_hs, hparams)
  # 1x1 Conv.
  continuous_hs = tf.layers.dense(continuous_hs, 512)
  # Relu.
  continuous_hs = tf.nn.relu(continuous_hs)
  # Layer Norm.
  continuous_hs = common_layers.layer_preprocess(continuous_hs, hparams)
  # 1x1 Conv.
  continuous_hs = tf.layers.dense(continuous_hs, 128)
  # Residul.
  continuous_hs += continuous_res
  # Continuous is now complete.

  # Clinical Notes Branch.
  clinical_res = clinical_notes
  # Layer Norm.
  clinical_hs = common_layers.layer_preprocess(clinical_notes, hparams)
  # Self Attention.
  with tf.variable_scope('clinical_attention'):
    clinical_hs = common_attention.multihead_attention(
        query_antecedent=clinical_hs,
        memory_antecedent=None,
        bias=None,
        total_key_depth=128,
        total_value_depth=128,
        output_depth=128,
        num_heads=8,
        dropout_rate=hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        max_relative_position=hparams.max_relative_position,
        dropout_broadcast_dims=dropout_broadcast_dims)
  # Residul.
  clinical_hs += clinical_res
  # Layer Norm.
  clinical_hs = common_layers.layer_preprocess(clinical_hs, hparams)
  clinical_res = clinical_hs
  # 1x1 Conv.
  clinical_hs = tf.layers.dense(clinical_hs, 512)
  # Relu.
  clinical_hs = tf.nn.relu(clinical_hs)
  # Layer Norm.
  clinical_hs = common_layers.layer_preprocess(clinical_hs, hparams)
  # 1x1 Conv.
  clinical_hs = tf.layers.dense(clinical_hs, 128)
  # Residul.
  clinical_hs += clinical_hs
  # Clinical is now complete.

  # Categorical Data Branch.
  categorical_res = categorical_data
  # Layer Norm.
  categorical_hs = common_layers.layer_preprocess(categorical_data, hparams)
  # Self Attention.
  with tf.variable_scope('categorical_attention'):
    categorical_hs = common_attention.multihead_attention(
        query_antecedent=categorical_hs,
        memory_antecedent=None,
        bias=None,
        total_key_depth=128,
        total_value_depth=128,
        output_depth=128,
        num_heads=8,
        dropout_rate=hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        max_relative_position=hparams.max_relative_position,
        dropout_broadcast_dims=dropout_broadcast_dims)
  # Concatenation.
  categorical_hs = tf.concat([categorical_hs, categorical_res], axis=-1)
  categorical_res = categorical_hs
  # 1x1 Conv.
  categorical_hs = tf.layers.dense(categorical_hs, 512)
  # Relu.
  categorical_hs = tf.nn.relu(categorical_hs)
  # Right Path - Layer Norm.
  categorical_res = common_layers.layer_preprocess(categorical_res, hparams)
  # Hybrid Fusion Point.
  categorical_hybrid_point = categorical_hs + tf.pad(categorical_res,
                                                     [[0, 0], [0, 0], [0, 256]])
  # Right Path - Dense.
  categorical_res = tf.layers.dense(categorical_res, 512)
  # Right Path - Relu. Late Fusion Point.
  categorical_late_point = tf.nn.relu(categorical_res)

  # Fusion Architecture.
  fusion_hs = tf.concat([clinical_hs, categorical_hybrid_point], axis=-1)
  fusion_res = fusion_hs
  # Layer Norm.
  fusion_hs = common_layers.layer_preprocess(fusion_hs, hparams)
  # Side Branch.
  separable_conv_1d = tf.layers.SeparableConv1D(
      384, 3, name='separable_conv_3x1', padding='SAME')
  fusion_sepconv_branch = separable_conv_1d.apply(fusion_hs)
  # Dense.
  fusion_hs = tf.layers.dense(fusion_hs, 1536)
  # Relu.
  fusion_hs = tf.nn.relu(fusion_hs)
  # Dense.
  fusion_hs = tf.layers.dense(fusion_hs, 384)

  output = fusion_res + tf.pad(fusion_hs, [[0, 0], [0, 0], [0, 256]]) + tf.pad(
      fusion_sepconv_branch, [[0, 0], [0, 0], [0, 256]]) + tf.pad(
          categorical_late_point, [[0, 0], [0, 0], [0, 128]]) + tf.pad(
              continuous_hs, [[0, 0], [0, 0], [0, 512]])

  output = common_layers.layer_preprocess(output, hparams)
  return output


class MultimodalTransformerModel(object):
  """Multimodal Transformer model."""

  def create_model_hparams(self):
    return hparam.HParams(
        batch_size=2,
        clip_norm=1.0,
        hidden_size=256,
        use_padding=False,
        hidden_layer_dim=32,
        hidden_layer_keep_prob=1.0,
        output_bias=0.0,

        # All this stuff is internal to tensor2tensor transformer layer.
        ffn_layer='dense_relu_dense',
        layer_postprocess_sequence='da',
        attention_dropout=.25,
        block_length=40,
        max_length=256,
        num_encoder_layers=2,
        num_heads=2,
        num_hidden_layers=1,
        pos='',
        proximity_bias=False,
        self_attention_type='dot_product',
        unidirectional_encoder=True,
        use_target_space_embedding=False,

        # All this stuff is internal to tensor2tensor learning rate schedule.
        learning_rate_decay_rate=.96,
        learning_rate_decay_staircase=False,
        learning_rate_decay_steps=4000,
        learning_rate_schedule='constant',
        learning_rate_warmup_steps=6000,
        train_steps=10,
    )

  def create_hparams(self, hparams_overrides=None):
    """Returns hyperparameters, including any flag value overrides.

    In order to allow for automated hyperparameter tuning, model hyperparameters
    are aggregated within a tf.HParams object.
    If the hparams_override FLAG is set, then it will use any values specified
    in hparams_override to override any individually-set hyperparameter.
    This logic allows tuners to override hyperparameter settings to find optimal
    values.

    Args:
      hparams_overrides: Hparams overriding existing ones. The priority of
        conflicting hparams: Lowest are the defaults defined in the model's
        create_params() function, then the ones from the file from hparam_path,
        and then the ones from this arg (e.g. through vizier),
        and highest priority are the flag hparams.

    Returns:
      The hyperparameters as a tf.HParams object.
    """
    hparams = self.create_model_hparams()
    if hparams_overrides:
      hparams = tf.training.merge_hparam(hparams, hparams_overrides)
    return hparams

  def reduce_sequence(self, hparams, all_sequence_embeddings_list,
                      sequence_length):
    """Process sequence inpus."""

    params = transformer.transformer_base_v2()
    for k, v in six.iteritems(hparams.values()):
      if k in params:
        params.set_hparam(k, v)
      else:
        params.add_hparam(k, v)

    # Here we project input feature dimensions into the customized dimensions.
    # It is optional.
    hidden_size_for_all_modalities = [8, 16, 32]
    assert len(hidden_size_for_all_modalities) == len(
        all_sequence_embeddings_list)
    all_sequence_embeddings_list = [
        tf.keras.layers.Conv1D(
            filters=single_hidden_size,
            padding='valid',
            kernel_size=1,
        )(sequence_embeddings)
        for single_hidden_size, sequence_embeddings in zip(
            hidden_size_for_all_modalities, all_sequence_embeddings_list)
    ]
    tf.logging.info('Projected all_sequence_embeddings_list:\n%s',
                    all_sequence_embeddings_list)

    encoder_output = mufasa_model(
        input_tensors=all_sequence_embeddings_list,
        hparams=params)

    # sequence_length is a vector containing sequence lengths in a batch.
    indices = tf.range(tf.cast(tf.shape(sequence_length)[0], dtype=tf.int64))
    indices = tf.stack([indices, sequence_length - 1], axis=1)
    last_output = tf.gather_nd(encoder_output, indices)
    return encoder_output, last_output

  def get_2d_and_3d_prelogits(self, hparams, features, mode):
    """Returns processed features and feature_columns according to hparams.

    Args:
      hparams: The hyperparameters.
      features: A dictionary of tensors keyed by the feature name.
      mode: The execution mode, as defined in tf.estimator.ModeKeys.

    Returns:
      A tuple (2d_prelogits, 3d_prelogits).
      - 2d_prelogits is the prelogits based on the last sequence output combined
        with context features.
      - 3d_prelogits is the prelogits at each bag in the sequence.
    """
    # This is the simulated data in the shape [batch_size, seq_len, hidden_dim].
    # Here we assume dimensions [0:8] are for categorical features;
    # Dimensions [8:24] are for continuous feature; Dimensions [24:56] are
    # for the clinical notes. They are for the exemplified purpose.
    all_sequence_embeddings = [
        features[:, :, 0:8], features[:, :, 8:24], features[:, :, 24:56]
    ]
    # Here we assume all sequences have the same sequence length, which is 10.
    # In the real EHR data, the sequence length could be very different for
    # different patients. We need to do paddings.
    sequence_length = tf.constant(hparams.batch_size * [10], dtype=tf.int64)
    seq_output, last_output = self.reduce_sequence(
        hparams, all_sequence_embeddings, sequence_length)

    # 4. Combine the results of the sequence model with context features.
    context_output = tf.random.normal([hparams.batch_size, 2])
    combined_features = tf.concat(
        [last_output, context_output],
        axis=-1)

    # 5. Maybe add a final hidden layer with optional dropout.
    if hparams.hidden_layer_dim > 0:
      combined_features = tf.layers.dense(
          combined_features, hparams.hidden_layer_dim, activation=tf.nn.relu6)
      if hparams.hidden_layer_keep_prob < 1.0:
        combined_features = tf.layers.dropout(
            combined_features,
            rate=1.0 - hparams.hidden_layer_keep_prob,
            training=(mode == tf.estimator.ModeKeys.TRAIN))

    return combined_features, seq_output

  def create_logits_fn(self, hparams, logits_dimension=1):
    """Return logits function."""

    def logits_fn(features, mode):
      """Creates the logits.

      Args:
        features: A dictionary of tensors keyed by the feature name.
        mode: The execution mode, as defined in tf.estimator.ModeKeys.

      Returns:
        logits: 2d logits based on the last sequence output combined with
          context features.
      """
      prelogits_2d, _ = self.get_2d_and_3d_prelogits(hparams, features, mode)
      return tf.layers.dense(
          prelogits_2d,
          logits_dimension,
          bias_initializer=tf.constant_initializer(hparams.output_bias))

    return logits_fn

  def create_model_fn(self, hparams):
    """Returns a function to build the model.

    Args:
      hparams: The hyperparameters.

    Returns:
      A function to build the model's graph. This function is called by
      the Estimator object to construct the graph.
    """
    logits_fn = self.create_logits_fn(hparams, 1)

    def model_fn(features, labels, mode):
      logits = logits_fn(features, mode)
      optimizer = tf.train.AdamOptimizer(
          learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
      return self.create_estimator_spec(
          logits=logits,
          labels=labels,
          mode=mode,
          optimizer=optimizer)
    return model_fn

  def _compute_loss(self, labels, predictions):
    """Computes the cross entropy loss.

    Args:
      labels: Tensor of shape [batch_size, logits_dimension] target integer
        labels in {0,1}.
      predictions: Dictionary of tensors with model predictions needed to
        compute metrics. Keys must be in constants.PredictionKeys. Particularly,
        predictions[constants.PredictionKeys.LOGITS] is a a Tensor of shape
        [batch_size, logits_dimension] with logits produced by the model.

    Returns:
      A Tensor of the same shape as labels with cross entropy loss.
    """
    logits = predictions['logits']
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)

  def _create_prediction_dict(self, logits):
    """Creates the prediction dict from logits.

    Args:
      logits: Tensor of shape [batch_size, logits_dimension] with logits
        produced by the model.

    Returns:
      A dict of Tensor that maps constants.PredictionKeys to the correspoinding
      predictions.
    """
    probabilities = tf.nn.softmax(logits)
    classes = tf.argmax(probabilities, axis=1)
    predictions = {
        'logits': logits,
        'probabilities': probabilities,
        'classes': classes,
    }
    return predictions

  def create_estimator_spec(self,
                            logits,
                            labels,
                            mode,
                            mask=None,
                            loss=None,
                            optimizer=None,
                            extra_eval_metrics=None):
    """Create EstimatorSpec for the prediction task.

    Args:
      logits: Tensor of shape [batch_size, logits_dimension].
      labels: A dict of dense label tensors. The key is label_key. The value is
        a tensor of shape [batch_size, logits_dimension] with 0/1 class labels
        for classification tasks, or shape [batch_size, 1] with numeric labels
        for regression.
      mode: The execution mode, as defined in tf.estimator.ModeKeys.
      mask: Tensor of type float32 that will be multiplied with the loss.
      loss: If provided, loss Tensor with shape [batch_size, 1]. If not
        provided, sigmoid cross entropy is used with given logits and labels.
      optimizer: tf.Optimizer instance. If not provided, defaults to Adam.
      extra_eval_metrics: If provided, a dictionary of `"name": tf.metrics.*
        object` entries to add to the existing evaluation metrics.

    Returns:
      EstimatorSpec with the mode, prediction, loss, train_op and
      export_outputs a dictionary specifying the output for a
      servo request during serving.
    """
    predictions = self._create_prediction_dict(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
      loss = None
    else:
      if loss is None:
        label_tensor = labels
        loss = self._compute_loss(label_tensor, predictions)
      if mask is not None:
        loss *= mask
      loss = tf.reduce_mean(loss)
      regularization_losses = tf.losses.get_regularization_losses()
      if regularization_losses:
        tf.summary.scalar('loss/prior_regularization', loss)
        regularization_loss = tf.add_n(regularization_losses)
        tf.summary.scalar('loss/regularization_loss', regularization_loss)
        loss += regularization_loss
      tf.summary.scalar('loss/train/', loss)

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

    if not extra_eval_metrics:
      eval_metric_ops = {}
    else:
      eval_metric_ops = extra_eval_metrics
    if mode == tf.estimator.ModeKeys.EVAL:
      label_tensor = labels
      eval_metric_ops['accuracy'] = self.accuracy_fn(label_tensor, predictions)
    merged = tf.summary.merge_all()
    training_hooks = []
    if merged is not None:
      training_hooks.append(
          tf.train.SummarySaverHook(save_steps=100, summary_op=merged))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        training_hooks=training_hooks)

  def accuracy_fn(self, labels, predictions):
    """Helper function for calculating the accuracy on uncensored examples."""
    return tf.metrics.accuracy(
        labels=labels, predictions=predictions['classes'])
