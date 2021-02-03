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

"""Head class for multi-event survival analysis."""

import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.head import multi_head


def _no_op_train_fn(loss):
  del loss
  return control_flow_ops.no_op()


def _default_export_output(export_outputs, head_name):
  """Extracts the default export output from the given export_outputs dict."""
  if len(export_outputs) == 1:
    return next(six.itervalues(export_outputs))
  try:
    return export_outputs[base_head.DEFAULT_SERVING_KEY]
  except KeyError:
    raise ValueError(
        '{} did not specify default export_outputs. '
        'Given: {} '
        'Suggested fix: Use one of the heads in tf.estimator, or include '
        'key {} in export_outputs.'.format(head_name, export_outputs,
                                           base_head.DEFAULT_SERVING_KEY))


def _get_per_head_label(labels):
  """Merge survival event labels and return a map keyed by head name.

    For classification labels, the map values are label values.
    For survival labels, the label values are maps with keys as [event_name] and
      '[event_name].time_to_event'.

  Args:
    labels: Dict keyed by 'event_name' and 'event_name.time_of_event' with value
      as tensors of shape [batch_size, 1].

  Returns:
    per_head_label_map: keyed by head name with value as label values.
      For survival labels, the value is a dict holding both event censor tensor
      and time_to_event tensor. For classification labels, the value is the
      label class. The tensor shapes are both [batch_size, 1].
  """
  per_head_label_map = {}
  for key in labels:
    if key + '.time_to_event' in labels:
      # key is a survival label.
      per_head_label_map[key] = dict()
      if len(labels[key].shape) == 1:
        raise ValueError(
            'expected_labels_shape: [batch_size,1]. labels_shape: {}.'.format(
                labels[key].shape))
      per_head_label_map[key][key] = labels[key]
      per_head_label_map[key][key + '.time_to_event'] = tf.cast(
          labels[key + '.time_to_event'], dtype=tf.float32)
    elif key.endswith('.time_to_event'):
      # this label is a time_to_event label, which is paired with the event.
      continue
    else:
      # key is a non-survival label.
      per_head_label_map[key] = labels[key]
  return per_head_label_map


class SurvivalMultiHead(multi_head.MultiHead):
  """MultiHead with support for survival head."""

  def loss(self,
           logits,
           labels,
           features=None,
           mode=None,
           regularization_losses=None):
    """Returns training loss. See `multi_head.MultiHead` for details."""
    logits_dict = self._check_logits_and_labels(logits, labels)
    per_head_label_map = _get_per_head_label(labels)

    training_losses = []
    for head in self._heads:
      # Each head is either a classification task or an event in survival
      # analysis.
      tf.logging.info(head.name)
      tf.logging.info(per_head_label_map[head.name])
      training_loss = head.loss(
          logits=logits_dict[head.name],
          labels=per_head_label_map[head.name],
          features=features,
          mode=mode)
      training_losses.append(training_loss)

    training_losses = tuple(training_losses)
    with ops.name_scope(
        'merge_losses',
        values=training_losses + (self._head_weights or tuple())):
      if self._head_weights:
        head_weighted_training_losses = []
        for training_loss, head_weight in zip(training_losses,
                                              self._head_weights):
          head_weighted_training_losses.append(
              math_ops.multiply(training_loss, head_weight))
        training_losses = head_weighted_training_losses
      merged_training_loss = math_ops.add_n(training_losses)

      if regularization_losses is None:
        regularization_loss = None
      else:
        regularization_loss = math_ops.add_n(regularization_losses)
      if regularization_loss is None:
        regularized_training_loss = merged_training_loss
      else:
        regularized_training_loss = merged_training_loss + regularization_loss
    return regularized_training_loss

  def update_metrics(self,
                     eval_metrics,
                     features,
                     logits,
                     labels,
                     regularization_losses=None):
    """Updates eval metrics. See `multi_head.MultiHead` for details."""
    logits_dict = self._check_logits_and_labels(logits, labels)

    per_head_label_map = _get_per_head_label(labels)

    # Update regularization loss metric
    if regularization_losses is not None:
      regularization_loss = math_ops.add_n(regularization_losses)
      eval_metrics[self._loss_regularization_key].update_state(
          values=regularization_loss)
    # Update metrics for each head
    for i, head in enumerate(self._heads):
      head_logits = logits_dict[head.name]
      head_labels = per_head_label_map[head.name]
      # Update loss metrics
      training_loss = head.loss(
          logits=head_logits, labels=head_labels, features=features)
      eval_metrics[self._loss_keys[i]].update_state(values=training_loss)
      # Update existing metrics in each head
      head_metrics = head.metrics()
      updated_metrics = head.update_metrics(head_metrics, features, head_logits,
                                            head_labels)
      eval_metrics.update(updated_metrics or {})
    return eval_metrics

  def create_estimator_spec(self,
                            features,
                            mode,
                            logits,
                            labels=None,
                            optimizer=None,
                            trainable_variables=None,
                            train_op_fn=None,
                            update_ops=None,
                            regularization_losses=None):
    """Returns a `model_fn.EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: Input `dict` keyed by head name, or logits `Tensor` with shape
        `[D0, D1, ... DN, logits_dimension]`. For many applications, the
        `Tensor` shape is `[batch_size, logits_dimension]`. If logits is a
        `Tensor`, it  will split the `Tensor` along the last dimension and
        distribute it appropriately among the heads. Check `MultiHead` for
        examples.
      labels: Input `dict` keyed by head name. For each head, the label value
        can be integer or string `Tensor` with shape matching its corresponding
        `logits`.`labels` is a required argument when `mode` equals `TRAIN` or
        `EVAL`.
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
        usually expressed as a batch average, so for best results, in each head,
        users need to use the default `loss_reduction=SUM_OVER_BATCH_SIZE` or
        set `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` to avoid scaling errors.
        Compared to the regularization losses for each head, this loss is to
        regularize the merged loss of all heads in multi head, and will be added
        to the overall training loss of multi head.

    Returns:
      A `model_fn.EstimatorSpec` instance.

    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
      mode, or if both are set.
      If `mode` is not in Estimator's `ModeKeys`.
    """
    per_head_label_map = _get_per_head_label(labels) if labels else None

    with ops.name_scope(self.name, 'multi_survival_head'):
      logits_dict = self._check_logits_and_labels(logits, labels)
      # Get all estimator spec.
      all_estimator_spec = []
      for head in self._heads:
        tf.logging.info(head.name)
        all_estimator_spec.append(
            head.create_estimator_spec(
                features=features,
                mode=mode,
                logits=logits_dict[head.name],
                labels=per_head_label_map[head.name] if labels else None,
                train_op_fn=_no_op_train_fn))
      # Predict.
      predictions = self.predictions(logits)
      if mode == model_fn.ModeKeys.PREDICT:
        export_outputs = self._merge_predict_export_outputs(all_estimator_spec)
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=export_outputs)
      loss = self.loss(logits, labels, features, mode, regularization_losses)
      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        eval_metrics = self.metrics(regularization_losses=regularization_losses)
        updated_metrics = self.update_metrics(
            eval_metrics,
            features,
            logits,
            labels,
            regularization_losses=regularization_losses)
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=updated_metrics)
      # Train.
      if mode == model_fn.ModeKeys.TRAIN:
        # train_op.
        if optimizer is not None:
          if train_op_fn is not None:
            raise ValueError('train_op_fn and optimizer cannot both be set.')
          if isinstance(optimizer, optimizer_v2.OptimizerV2):
            base_head.validate_trainable_variables(trainable_variables)
            train_op = optimizer.get_updates(
                loss, trainable_variables)
          else:
            train_op = optimizer.minimize(
                loss,
                global_step=training_util.get_global_step())
        elif train_op_fn is not None:
          train_op = train_op_fn(loss)
        else:
          raise ValueError('train_op_fn and optimizer cannot both be None.')
        # Create summary.
        base_head.create_estimator_spec_summary(loss, regularization_losses)
        # eval_metrics.
        eval_metrics = {}
        for spec in all_estimator_spec:
          eval_metrics.update(spec.eval_metric_ops or {})
        # predictions can be used to access the logits in `TRAIN` mode
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op,
            predictions=predictions,
            eval_metric_ops=eval_metrics)
      raise ValueError('mode={} unrecognized'.format(mode))
