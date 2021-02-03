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

"""Estimator heads for sequence labels."""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import metrics
from tensorflow.python.ops import math_ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head

MEAN_ABS_ERROR = 'mean_abs_error'
CONTEXT_WINDOW_SIZE = 'context_window_size'
STATE_SPACE_MODELS = ['dkf_ds']


class ObservationHead(base_head.Head):
  """Head with customized loss function and metrics for sequence labels."""

  def __init__(self,
               weight_column=None,
               label_dimension=1,
               loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
               name=None,
               model_hparams=None):
    self._weight_column = weight_column
    self._logits_dimension = label_dimension
    tf.logging.info(self._logits_dimension)
    self._loss_reduction = loss_reduction
    self._name = name
    self._model_hparams = model_hparams
    tf.logging.info(self._model_hparams)
    # Metric keys.
    keys = metric_keys.MetricKeys
    self._loss_mean_key = self._summary_key(keys.LOSS_MEAN)
    self._prediction_mean_key = self._summary_key(keys.PREDICTION_MEAN)
    self._label_mean_key = self._summary_key(keys.LABEL_MEAN)
    self._loss_regularization_key = self._summary_key(keys.LOSS_REGULARIZATION)
    self._mean_abs_error = self._summary_key(MEAN_ABS_ERROR)

  @property
  def name(self):
    """See `base_head.Head` for details."""
    return self._name

  @property
  def logits_dimension(self):
    """See `base_head.Head` for details."""
    return self._logits_dimension

  @property
  def loss_reduction(self):
    """See `base_head.Head` for details."""
    return self._loss_reduction

  def _processed_labels(self, logits, labels, features):
    """Converts labels to ground truth observation sequence.

    Args:
      logits: estimated obs. value, [batch, time_len, num_obs] tensor.
      labels: dict with obs and interv. code as keys and tensor with shape
        [batch, context_window_size] as value.
      features: dict from feature names to tensors.

    Returns:
      tensor for obs with shape [batch, context_window_size -1, num_obs], which
      rolls the input obs left for one time slot. Note that the last obs label
      comes from the first obs, which is invalid as label.
    """
    del logits
    if labels is None:
      raise ValueError(base_head._LABEL_NONE_ERR_MSG)  # pylint: disable=protected-access
    obs_tensor = tf.concat([
        tf.expand_dims(labels[obs], 2)
        for obs in self._model_hparams.observation_codes
    ],
                           axis=2)
    # Shift the label observation tensor towards left for one time slot.
    return tf.roll(obs_tensor, shift=-1, axis=1)

  def _unweighted_loss_and_weights(self, logits, processed_labels, features,
                                   mode):
    """Computes loss spec."""
    if self._model_hparams.model_name in STATE_SPACE_MODELS:
      _, loss = logits
      return loss, 1

    if self._model_hparams.has_mask:
      mask_obs_tensor = tf.concat([
          tf.expand_dims(features[obs + '_mask'], 2)
          for obs in self._model_hparams.observation_codes
      ],
                                  axis=2)
    else:
      mask_obs_tensor = tf.ones_like(processed_labels)

    if self._model_hparams.has_mask:
      true_len_hr = features['true_length_hr']
      seqlen = tf.reduce_max(true_len_hr)
    else:
      batch_size = logits.get_shape().as_list()[0]
      true_len_hr = tf.fill([batch_size],
                            self._model_hparams.context_window_size)
      seqlen = self._model_hparams.context_window_size
    if self._model_hparams.model_name == 'lstm_ds':
      seqlen = seqlen - 1

    # processed_labels is not trimmed with true len with
    # [batch, context_window_size -1, num_obs]
    # logit is trimmed with seqlen with shape [batch, seqlen, num_obs]
    # batch_time_value_loss shape [batch, seqlen]
    # seqlen is #values, if seqlen = 5, there are 5 values for each feature.
    # trimmed_processed_labels shape [batch, seqlen, num_obs]
    trimmed_processed_labels = tf.slice(processed_labels, [0, 0, 0],
                                        [-1, seqlen, -1])
    trimmed_mask_obs_tensor = tf.slice(mask_obs_tensor, [0, 0, 0],
                                       [-1, seqlen, -1])

    # Compute L2 value loss based on true sequence len/windown size and obs val.
    batch_time_feature_loss = tf.multiply(
        tf.square(trimmed_processed_labels - logits), trimmed_mask_obs_tensor)
    batch_time_value_loss = tf.reduce_mean(batch_time_feature_loss, axis=2)

    # self._model_hparams.last_obs_len is the num of most recent observations
    # used for computing loss.
    # batch_value_loss shape [batch, 1].
    last_obs_len = self._model_hparams.last_obs_len
    assert last_obs_len < self._model_hparams.context_window_size

    # zero out the loss outside [seqlen-last_obs_len, seqlen].
    true_len_mask = tf.sequence_mask(true_len_hr, seqlen)
    if last_obs_len == -1:
      # all obs up to true len are included.
      selection_mask = tf.cast(true_len_mask, tf.float32)
    else:
      last_obs_len_mask = tf.sequence_mask(true_len_hr - last_obs_len, seqlen)
      selection_mask = tf.cast(
          tf.logical_xor(true_len_mask, last_obs_len_mask), tf.float32)

    trimmed_batch_time_value_loss = tf.multiply(batch_time_value_loss,
                                                selection_mask)
    batch_value_loss = tf.div_no_nan(
        tf.reduce_sum(trimmed_batch_time_value_loss, axis=1),
        tf.reduce_sum(selection_mask, axis=1))

    scalar_loss = math_ops.reduce_mean(batch_value_loss, axis=-1, keepdims=True)

    weights = base_head.get_weights_and_check_match_logits(
        features=features, weight_column=self._weight_column, logits=logits)
    return scalar_loss, weights

  def loss(self,
           logits,
           labels,
           features=None,
           mode=None,
           regularization_losses=None):
    """Compute Loss.

    Args:
      logits: estimated obs. value, [batch, context_window_size, num_obs]
        tensor.
      labels: ground truth observation, feature dict with obs. and interv. codes
        as keys, values tensor with shape [batch_size, context_window_size].
      features: dict with feature name strings as key, tensor as value.
      mode: see base_head.Head.
      regularization_losses: see base_head.Head.

    Returns:
      regularized_training_loss: see base_head.Head.
    """
    with ops.name_scope(
        'losses', values=(logits, labels, regularization_losses, features)):
      # processed_labels is a [batch, time_len, num_obs] tensor.
      processed_labels = self._processed_labels(logits, labels, features)
      unweighted_loss, weights = self._unweighted_loss_and_weights(
          logits, processed_labels, features, mode)
      training_loss = tf.losses.compute_weighted_loss(
          unweighted_loss, weights=weights, reduction=self._loss_reduction)
      if regularization_losses is None:
        regularization_loss = None
      else:
        regularization_loss = math_ops.add_n(regularization_losses)
      if regularization_loss is None:
        regularized_training_loss = training_loss
      else:
        regularized_training_loss = training_loss + regularization_loss

    return regularized_training_loss

  def predictions(self, logits, keys=None):
    """Return predictions based on keys.

    Args:
      logits: estimated obs. value, [batch, time_len, num_obs] tensor.
      keys: a list of prediction keys. Key can be either the class variable
        of prediction_keys.PredictionKeys or its string value, such as:
          prediction_keys.PredictionKeys.LOGITS or 'logits'.

    Returns:
      predictions: the predicted values for the last_obs_len number of obs.
    """
    pred_keys = prediction_keys.PredictionKeys
    valid_keys = [pred_keys.LOGITS, pred_keys.PROBABILITIES]
    if keys:
      base_head.check_prediction_keys(keys, valid_keys)
    else:
      keys = valid_keys
    # pred shape [bs, self._config.context_window_size, self._a_dim]
    if self._model_hparams.model_name in STATE_SPACE_MODELS:
      obs_est, _ = logits
    else:
      obs_est = logits
    predictions = {}
    with ops.name_scope('predictions', values=(logits,)):
      if pred_keys.LOGITS in keys:
        predictions[pred_keys.LOGITS] = obs_est[:, (
            -1 - self._model_hparams.last_obs_len):-1, :]
      return predictions

  def metrics(self, regularization_losses=None):
    """Creates metrics. See `base_head.Head` for details."""
    keys = metric_keys.MetricKeys
    with ops.name_scope('metrics', values=(regularization_losses,)):
      # Mean metric.
      eval_metrics = {}
      eval_metrics[self._loss_mean_key] = metrics.Mean(name=keys.LOSS_MEAN)
      eval_metrics[self._prediction_mean_key] = metrics.Mean(
          name=keys.PREDICTION_MEAN)
      eval_metrics[self._label_mean_key] = metrics.Mean(name=keys.LABEL_MEAN)
      if regularization_losses is not None:
        eval_metrics[self._loss_regularization_key] = metrics.Mean(
            name=keys.LOSS_REGULARIZATION)
      eval_metrics[self._mean_abs_error] = metrics.MeanAbsoluteError(
          name=MEAN_ABS_ERROR, dtype=tf.float32)

    return eval_metrics

  def update_metrics(self,
                     eval_metrics,
                     features,
                     logits,
                     labels,
                     regularization_losses=None):
    """Updates eval metrics.

    Args:
      eval_metrics: See `base_head.Head` for details.
      features: dict with feature name strings as key, tensor as value.
      logits: estimated obs. value, [batch, time_len, num_obs] tensor.
      labels: ground truth observation, feature dict with obs. and interv. codes
        as keys, values tensor with shape [batch_size, context_window_size].
      regularization_losses: See `base_head.Head` for details.

    Returns:
      eval_metrics: See `base_head.Head` for details.
    """
    processed_labels = self._processed_labels(logits, labels, features)
    unweighted_loss, weights = self._unweighted_loss_and_weights(
        logits, processed_labels, features, mode=model_fn.ModeKeys.EVAL)
    # Update metrics.
    eval_metrics[self._loss_mean_key].update_state(
        values=unweighted_loss, sample_weight=weights)
    value_key = prediction_keys.PredictionKeys.LOGITS
    predictions = self.predictions(logits, [value_key])
    value_predictions = predictions[value_key]

    base_head.update_metric_with_broadcast_weights(
        eval_metrics[self._prediction_mean_key], value_predictions, weights)
    eval_metrics[self._mean_abs_error].update_state(
        processed_labels[:, (-1 - self._model_hparams.last_obs_len):-1, :],
        value_predictions)

    # label_mean represents the percentage of censored events. In case of
    # mortality, it is the percentage of survived patients.
    base_head.update_metric_with_broadcast_weights(
        eval_metrics[self._label_mean_key], processed_labels, weights)

    if regularization_losses is not None:
      regularization_loss = math_ops.add_n(regularization_losses)
      eval_metrics[self._loss_regularization_key].update_state(
          values=regularization_loss)
    return eval_metrics

  def _create_tpu_estimator_spec(self,
                                 features,
                                 mode,
                                 logits,
                                 labels=None,
                                 optimizer=None,
                                 trainable_variables=None,
                                 train_op_fn=None,
                                 update_ops=None,
                                 regularization_losses=None):
    """Returns an `model_fn._TPUEstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: estimated obs. value, [batch, time_len, num_obs] tensor.
      labels: ground truth observation, feature dict with obs. and interv. codes
        as keys, values tensor with shape [batch_size, context_window_size].
      optimizer: An `tf.keras.optimizers.Optimizer` instance to optimize the
        loss in TRAIN mode. Namely, sets `train_op = optimizer.get_updates(loss,
        trainable_variables)`, which updates variables to minimize `loss`.
      trainable_variables: A list or tuple of `Variable` objects to update to
        minimize `loss`. In Tensorflow 1.x, by default these are the list of
        variables collected in the graph under the key
        `GraphKeys.TRAINABLE_VARIABLES`. As Tensorflow 2.x doesn't have
        collections and GraphKeys, trainable_variables need to be passed
        explicitly here.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      update_ops: A list or tuple of update ops to be run at training time. For
        example, layers such as BatchNormalization create mean and variance
        update ops that need to be run at training time. In Tensorflow 1.x,
        these are thrown into an UPDATE_OPS collection. As Tensorflow 2.x
        doesn't have collections, update_ops need to be passed explicitly here.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.

    Returns:
      `model_fn._TPUEstimatorSpec`.
    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
    with ops.name_scope(self._name, 'sequence_head'):
      # Predict.
      predictions = self.predictions(logits)
      if mode == model_fn.ModeKeys.PREDICT:
        return model_fn._TPUEstimatorSpec(  # pylint:disable=protected-access
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                base_head.DEFAULT_SERVING_KEY:
                    export_output.PredictOutput(predictions),
                base_head.PREDICT_SERVING_KEY: (
                    export_output.PredictOutput(predictions))
            })

      regularized_training_loss = self.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=mode,
          regularization_losses=regularization_losses)
      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        eval_metrics = self.metrics(regularization_losses=regularization_losses)
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=regularized_training_loss,
            eval_metrics=base_head.create_eval_metrics_tuple(
                self.update_metrics, {
                    'eval_metrics': eval_metrics,
                    'features': features,
                    'logits': logits,
                    'labels': labels,
                    'regularization_losses': regularization_losses
                }))
      # Train.
      train_op = base_head.create_estimator_spec_train_op(
          self._name,
          optimizer=optimizer,
          trainable_variables=trainable_variables,
          train_op_fn=train_op_fn,
          update_ops=update_ops,
          regularized_training_loss=regularized_training_loss)
    # Create summary.
    base_head.create_estimator_spec_summary(regularized_training_loss,
                                            regularization_losses,
                                            self._summary_key)
    return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op)
