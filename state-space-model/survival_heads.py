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

"""Estimator head for survival analysis."""
import survival_util
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import metrics
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.saved_model import signature_def_utils
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head


SURVIVAL_SERVING_KEY = 'survival'
HAZARD_RATE = 'hazard_rate'
PREDICTED_TIME = 'predicted_time'
SLOT_TO_WINDOW = 24
UNITS_IN_HR = 60 * 60


class SurvivalOutput(export_output.ExportOutput):
  """Represents the output of a survival head."""

  def __init__(self, value):
    """Initializer for `SurvivalOutput`.

    Args:
      value: a float `Tensor` with shape [time_len] giving the hazard values.

    Raises:
      ValueError: if the value is not a `Tensor` with dtype tf.float32.
    """
    if not (isinstance(value, ops.Tensor) and value.dtype.is_floating):
      raise ValueError('Survival output value must be a float32 Tensor; '
                       'got {}'.format(value))
    self._value = value

  @property
  def value(self):
    return self._value

  def as_signature_def(self, receiver_tensors):
    if len(receiver_tensors) != 2:
      raise ValueError('Survival input must be dict with two entry; '
                       'got {}'.format(receiver_tensors))
    examples = receiver_tensors['input_examples']
    if dtypes.as_dtype(examples.dtype) != dtypes.string:
      raise ValueError('Survival input must be a single string Tensor; '
                       'got {}'.format(receiver_tensors))
    return signature_def_utils.regression_signature_def(examples, self.value)


PROBABILITY_AT_WINDOW = 'probability_at_window_%d'
AUC_PR = 'pr_auc_%s'
AUC_ROC = 'auc_roc_%s'
MEAN_ABS_ERROR = 'mean_abs_error_%s'


class SurvivalHead(base_head.Head):
  """Survival analysis head with customized loss function and metrics."""

  def __init__(self,
               weight_column=None,
               label_dimension=1,
               all_event_list=None,
               loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
               survival_model_name='exponential',
               name=None,
               model_hparams=None):
    self._weight_column = weight_column
    self._survival_model_name = survival_model_name
    tf.logging.info(survival_model_name)
    self._survival_model = survival_util.REGISTERED_SURVIVAL_MODEL[
        self._survival_model_name]
    # For independent events, label_dimension == 1
    # For correlated events, label_dimension == num_labels == num_events
    self._logits_dimension = label_dimension
    tf.logging.info(self._logits_dimension)
    self._loss_reduction = loss_reduction
    self._name = name
    self._model_hparams = model_hparams
    if all_event_list is None:
      # For single event, independent events.
      self._all_event_list = [name]
    else:
      self._all_event_list = all_event_list
    tf.logging.info(self._all_event_list)

    self._event_index = 0
    for i, event in enumerate(self._all_event_list):
      if event == self._name:
        self._event_index = i
        break
    tf.logging.info(self._name)
    tf.logging.info(self._event_index)
    # Metric keys.
    keys = metric_keys.MetricKeys
    self._loss_mean_key = self._summary_key(keys.LOSS_MEAN)
    self._prediction_mean_key = self._summary_key(keys.PREDICTION_MEAN)
    self._label_mean_key = self._summary_key(keys.LABEL_MEAN)
    self._loss_regularization_key = self._summary_key(keys.LOSS_REGULARIZATION)
    self._auc_roc_24 = self._summary_key(AUC_ROC % '24')
    self._auc_roc_48 = self._summary_key(AUC_ROC % '48')

    if self._model_hparams.da_tlen > 0:
      self._auc_pr = self._summary_key(AUC_PR % 'avg')
      self._auc_roc = self._summary_key(AUC_ROC % 'avg')
      self._mean_abs_error = self._summary_key(MEAN_ABS_ERROR % 'avg')

      self._probablity_within_window_list = []
      for i in range(int(self._model_hparams.da_tlen / SLOT_TO_WINDOW) + 1):
        self._probablity_within_window_list.append(
            self._summary_key(PROBABILITY_AT_WINDOW % i))

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

  def _processed_logits(self, logits):
    """Process logits.

    Args:
      logits: for single event or independent event, a tensor of shape
        [batch_size, 1]; for correlated event, a dict with event_name as key,
        value as tensor of shape [batch_size, 1].

    Returns:
      logits: tensor of shape [batch_size, logits_dimension]. For correlated
        event, logits_dimension == num_events, value arranged based on
        the order in _all_event_list; otherwise, logits_dimension == 1.
    """
    tf.logging.info(logits)
    if isinstance(logits, dict):
      all_event_logits_list = []
      for event in self._all_event_list:
        all_event_logits_list.append(logits[event])
      # shape of all_event_logits is [batch_size, num_events]
      all_event_logits = tf.concat(all_event_logits_list, axis=-1)
    else:
      all_event_logits = logits
    tf.logging.info(self.logits_dimension)
    tf.logging.info(all_event_logits)
    logits = base_head.check_logits_final_dim(all_event_logits,
                                              self.logits_dimension)

    return logits

  def _processed_labels(self, logits, labels):
    """Converts labels to time_to_event, ~censored.

    Args:
      logits: logits `Tensor` with shape `[batch_size, _logits_dimension]`.
      labels: dict with [event_name] and [event_name].time_to_event as keys. The
        value for [event_name] key is a Tensor of shape `[batch_size, 1]` of
        type int64, indicating whether the event is observed. The value for
        [event_name].time_to_event key is a Tensor of shape
        `[batch_size, 1]` of type float32. Here is one example label:
        {u'respiration_failure.time_to_event':
         <tf.Tensor 'Cast:0' shape=(32, 1) dtype=float32>,
         u'respiration_failure':
        <tf.Tensor 'Batch/batch:110' shape=(32, 1) dtype=int64>}

    Returns:
      Tuple of two tensors, time_to_event, event_censored.
      For single event or independent events, each tensor has shape
        [batch_size, 1]; for correlated events, each tensor has shape
        [batch_size, num_events], value arranged based on the order in
        _all_event_list.
    """
    del logits

    if labels is None:
      raise ValueError(base_head._LABEL_NONE_ERR_MSG)  # pylint:disable=protected-access
    if len(labels) > 2:
      # For correlated events, labels for all events are provided.
      tf.logging.info('correlated event labels')
    event_observed_dict = dict()
    time_to_event_dict = dict()
    for key in labels:
      if not key.endswith('.time_to_event'):
        event_observed_dict[key] = labels[key]
        if key + '.time_to_event' in labels:
          time_to_event_dict[key] = tf.cast(
              labels[key + '.time_to_event'], dtype=tf.float32)
        else:
          raise ValueError(key + '.time_to_event should be in labels.')

    event_observed_list = []
    time_to_event_list = []
    tf.logging.info(self._all_event_list)
    for event in self._all_event_list:
      event_observed_list.append(event_observed_dict[event])
      time_to_event_list.append(time_to_event_dict[event])

    # event_observed_list is a list of tensor with shape [batch_size, 1]
    tf.logging.info(event_observed_list)
    # event_observed has shape [batch_size, num_events] for correlated event.
    # event_observed has shape [batch_size, 1] for single/independent event.
    event_observed = tf.concat(event_observed_list, axis=-1)
    event_censored = tf.equal(event_observed, tf.constant([0], dtype=tf.int64))
    time_to_event = tf.concat(time_to_event_list, axis=-1)
    return time_to_event, event_censored

  def _unweighted_loss_and_weights(self, logits, processed_labels, features):
    """Computes loss spec."""
    time_to_event, censored = processed_labels

    time_to_event.shape.assert_is_compatible_with(censored.shape)
    with tf.control_dependencies([
        tf.assert_positive(time_to_event),
    ]):
      model = self._survival_model(
          params=logits,
          labels=processed_labels,
          event_index=self._event_index,
          model_hparams=self._model_hparams)

      tf.logging.info(model.params())
      log_pdf_value = model.log_pdf(time_to_event)
      log_survival_value = model.log_survival_func(time_to_event)
      batch_loss = survival_util.negative_log_likelihood_loss(
          censored=censored,
          log_pdf_value=log_pdf_value,
          log_survival_value=log_survival_value)
      # batch_loss has shape [batch_size,1]
      tf.logging.info(batch_loss)

      scalar_loss = math_ops.reduce_mean(batch_loss, axis=-1, keepdims=True)
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
      logits: for single event, indepdent event, logits is a tensor of shape
        [batch_size, 1]; for correlated event, a dict with event_name as key,
        value as tensor of shape [batch_size, 1].
      labels: dict keyed by 'event_name' and 'event_name.time_of_event' with
        value as tensors of shape [batch_size] or [batch_size, 1]. For
        correlated events, labels for all events are provided. Otherwise, only
        the event associated with this head is provided.
      features: see base_head.Head.
      mode: see base_head.Head.
      regularization_losses: see base_head.Head.

    Returns:
      regularized_training_loss: see base_head.Head.
    """

    del mode  # Unused for this head.
    with ops.name_scope(
        'losses', values=(logits, labels, regularization_losses, features)):
      tf.logging.info(logits)
      # processed_logits shape is [batch_size, num_events] for correlated events
      # or [batch_size, 1] for single event.
      processed_logits = self._processed_logits(logits)

      tf.logging.info(labels)
      # processed_labels is a tuple with two tensors of shape
      # [batch_size, num_events] for correlated events, or [batch_size, 1]
      # for single event.
      processed_labels = self._processed_labels(logits, labels)

      unweighted_loss, weights = self._unweighted_loss_and_weights(
          processed_logits, processed_labels, features)
      training_loss = losses.compute_weighted_loss(
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
    """Return predictions based on keys..

    Args:
      logits: for single event, indepdent event, logits is a tensor of shape
        [batch_size, 1]; for correlated event, a dict with event_name as key,
        value as tensor of shape [batch_size, 1].
      keys: a list of prediction keys. Key can be either the class variable
        of prediction_keys.PredictionKeys or its string value, such as:
          prediction_keys.PredictionKeys.LOGITS or 'logits'.

    Returns:
      predictions: see base_head.Head.
    """
    pred_keys = prediction_keys.PredictionKeys
    valid_keys = [pred_keys.LOGITS, pred_keys.PROBABILITIES]
    if keys:
      base_head.check_prediction_keys(keys, valid_keys)
    else:
      keys = valid_keys

    processed_logits = self._processed_logits(logits)
    predictions = {}
    with ops.name_scope('predictions', values=(processed_logits,)):
      if pred_keys.LOGITS in keys:
        predictions[pred_keys.LOGITS] = processed_logits
      if pred_keys.PROBABILITIES in keys:
        model = self._survival_model(
            params=processed_logits,
            labels=None,
            event_index=self._event_index,
            model_hparams=self._model_hparams)
        predictions[pred_keys.PROBABILITIES] = model.probability()
      predictions[HAZARD_RATE] = model.params()
      predictions[PREDICTED_TIME] = model.predicted_time()
      return predictions

  def hazard_rates(self, logits, keys=None):
    """Return hazard_rates based on keys..

    Args:
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      keys: a list of prediction keys. Key can be either the class variable
        of prediction_keys.PredictionKeys or its string value, such as:
          prediction_keys.PredictionKeys.LOGITS or 'logits'.

    Returns:
      hazard_rates: tensor of shape [batch_size, 1]
    """
    processed_logits = self._processed_logits(logits)
    with ops.name_scope('hazard_rates', values=(logits,)):
      model = self._survival_model(
          params=processed_logits,
          labels=None,
          event_index=self._event_index,
          model_hparams=self._model_hparams)
      hazard_rates = model.params()
      tf.logging.info(hazard_rates)
      return hazard_rates

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
      if self._model_hparams.da_tlen > 0 and not self._model_hparams.event_relation:
        eval_metrics[self._auc_roc_24] = metrics.AUC(name=AUC_ROC % '24')
        eval_metrics[self._auc_roc_48] = metrics.AUC(name=AUC_ROC % '48')
        eval_metrics[self._auc_pr] = metrics.AUC(
            curve='PR', name=AUC_PR % 'avg')
        eval_metrics[self._auc_roc] = metrics.AUC(name=AUC_ROC % 'avg')
        eval_metrics[self._mean_abs_error] = metrics.MeanAbsoluteError(
            name=MEAN_ABS_ERROR % 'avg')

        for i in range(int(self._model_hparams.da_tlen / SLOT_TO_WINDOW) + 1):
          eval_metrics[self._probablity_within_window_list[i]] = metrics.Mean(
              name=PROBABILITY_AT_WINDOW % i)

    return eval_metrics

  def _true_and_predict_within_window(self, time_to_event, censored,
                                      probabilities_at_window, window_start_t,
                                      window_end_t):
    time_to_event_within_window = tf.logical_and(
        tf.greater_equal(time_to_event, window_start_t),
        tf.less_equal(time_to_event, window_end_t))
    event_within_window = tf.logical_and(
        tf.logical_not(censored), time_to_event_within_window)
    # Excluded from computation.
    # censored_within_window shape [batch_size, 1].
    censored_within_window = tf.logical_and(censored,
                                            time_to_event_within_window)
    # Excluding the events that are censored within the window.
    # true/1, if the event is observed within the window.
    # false/0, if the event is observed or censored outside of the window.
    y_true = tf.boolean_mask(event_within_window,
                             tf.logical_not(censored_within_window))
    # probabilities_at_window shape [batch_size, 1]
    y_pred = tf.boolean_mask(probabilities_at_window,
                             tf.logical_not(censored_within_window))
    tf.logging.info(y_pred)
    if y_pred.shape.ndims > 1:
      y_pred = tf.squeeze(y_pred, axis=-1)
    return y_true, y_pred

  def update_metrics(self,
                     eval_metrics,
                     features,
                     logits,
                     labels,
                     regularization_losses=None):
    """Updates eval metrics.

    Args:
      eval_metrics: See `base_head.Head` for details.
      features: See `base_head.Head` for details.
      logits: for single event, indepdent event, logits is a tensor of shape
        [batch_size, 1], for correlated event, a dict with event_name as key,
        value as tensor of shape [batch_size, 1].
      labels: dict keyed by 'event_name' and 'event_name.time_of_event' with
        value as tensors of shape [batch_size] or [batch_size, 1]. For
        correlated events, labels for all events are provided. Otherwise, only
        the event associated with this head is provided.
      regularization_losses: See `base_head.Head` for details.

    Returns:
      eval_metrics: See `base_head.Head` for details.
    """
    processed_logits = self._processed_logits(logits)
    processed_labels = self._processed_labels(logits, labels)
    time_to_event, censored = processed_labels

    unweighted_loss, weights = self._unweighted_loss_and_weights(
        processed_logits, processed_labels, features)

    # Update metrics.
    eval_metrics[self._loss_mean_key].update_state(
        values=unweighted_loss, sample_weight=weights)
    prob_key = prediction_keys.PredictionKeys.PROBABILITIES
    predictions = self.predictions(logits, [prob_key])
    probabilities = predictions[prob_key]

    base_head.update_metric_with_broadcast_weights(
        eval_metrics[self._prediction_mean_key], probabilities, weights)

    if self._model_hparams.da_tlen > 0 and not self._model_hparams.event_relation:
      y_true_list = []
      y_pred_list = []
      for i in range(int(self._model_hparams.da_tlen / SLOT_TO_WINDOW) + 1):
        model = self._survival_model(
            params=processed_logits,
            labels=processed_labels,
            event_index=self._event_index,
            model_hparams=self._model_hparams)
        window_start_t = i * UNITS_IN_HR * self._model_hparams.da_sslot * SLOT_TO_WINDOW  # pylint: disable=line-too-long
        window_end_t = (
            i + 1) * UNITS_IN_HR * self._model_hparams.da_sslot * SLOT_TO_WINDOW
        # probabilities_at_window shape [batch_size, 1].
        probabilities_at_window = model.probability_within_window(
            window_start_t=window_start_t, window_end_t=window_end_t)
        base_head.update_metric_with_broadcast_weights(
            eval_metrics[self._probablity_within_window_list[i]],
            probabilities_at_window, weights)
        y_true, y_pred = self._true_and_predict_within_window(
            time_to_event, censored, probabilities_at_window, window_start_t,
            window_end_t)
        # y_true, y_pred shape [batch_size]
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        tf.logging.info(y_true)
        tf.logging.info(y_pred)

      eval_metrics[self._auc_pr].update_state(
          tf.concat(y_true_list, axis=0), tf.concat(y_pred_list, axis=0))
      eval_metrics[self._auc_roc].update_state(
          tf.concat(y_true_list, axis=0), tf.concat(y_pred_list, axis=0))

      # 24hr and 48hr window AUC.
      probabilities_at_24 = model.probability_within_window(
          window_start_t=0, window_end_t=UNITS_IN_HR * 24)
      y_true_24, y_pred_24 = self._true_and_predict_within_window(
          time_to_event, censored, probabilities_at_24, 0, UNITS_IN_HR * 24)
      eval_metrics[self._auc_roc_24].update_state(y_true_24, y_pred_24)

      probabilities_at_48 = model.probability_within_window(
          window_start_t=0, window_end_t=UNITS_IN_HR * 48)
      y_true_48, y_pred_48 = self._true_and_predict_within_window(
          time_to_event, censored, probabilities_at_48, 0, UNITS_IN_HR * 48)
      eval_metrics[self._auc_roc_48].update_state(y_true_48, y_pred_48)

      observed_time = tf.boolean_mask(
          time_to_event[:, self._event_index] / UNITS_IN_HR,
          tf.logical_not(censored[:, self._event_index]))

      predicted_time = model.predicted_time()
      predicted_time = tf.boolean_mask(
          predicted_time, tf.logical_not(censored[:, self._event_index]))
      eval_metrics[self._mean_abs_error].update_state(observed_time,
                                                      predicted_time)

    # label_mean represents the percentage of censored events. In case of
    # mortality, it is the percentage of survived patients.
    base_head.update_metric_with_broadcast_weights(
        eval_metrics[self._label_mean_key], censored, weights)

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
      logits: for single event, indepdent event, logits is a tensor of shape
        [batch_size, 1], for correlated event, a dict with event_name as key,
        value as tensor of shape [batch_size, 1].
      labels: dict keyed by 'event_name' and 'event_name.time_of_event' with
        value as tensors of shape [batch_size] or [batch_size, 1]. For
        correlated events, labels for all events are provided. Otherwise, only
        the event associated with this head is provided.
        Here is one example label:
        {u'respiration_failure.time_to_event':
         <tf.Tensor 'Cast:0' shape=(32,) dtype=float32>,
         u'respiration_failure':
        <tf.Tensor 'Batch/batch:110' shape=(32,) dtype=int64>} `labels` is
          required argument when `mode` equals `TRAIN` or `EVAL`.
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
    tf.logging.info(mode)

    with ops.name_scope(self._name, 'survival_head'):
      # Predict.
      predictions = self.predictions(logits)
      # hazard_rates = self.hazard_rates(logits)

      if mode == model_fn.ModeKeys.PREDICT:
        # survival_output = SurvivalOutput(value=hazard_rates)

        return model_fn._TPUEstimatorSpec(  # pylint:disable=protected-access
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                base_head.DEFAULT_SERVING_KEY: (
                    export_output.PredictOutput(predictions)),
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
    base_head.create_estimator_spec_summary(
        regularized_training_loss, regularization_losses, self._summary_key)
    return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op)
