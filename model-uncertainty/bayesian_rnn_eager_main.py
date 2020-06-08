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

"""Bayesian RNN train/eval/predict script.

This script contains custom training, evaluation, and prediction loops used for
the paper "Analyzing the Role of Model Uncertainty for Electronic Health
Records" (Dusenberry et al., 2020).

The code runs in Eager mode with TF 2.x and Python 3, and is intended to be used
as a "workbench" for rapid iteration in experimental work.

In general, this script can run training, evaluation, and prediction jobs,
settable via the "job" flag. By default, it runs a single job that runs a
training loop with evaluation on training and validation data periodically. Job
restarts and simultaneous jobs (such as separate, parallel training and
evaluation jobs) that start from previous training state are supported by
running the script again with the same "logdir" and "model_dir" flags. By
default, those flags use the "timestamp" flag as a suffix, which itself defaults
to a current date-time stamp. This latter "timestamp" flag can thus be used as a
convenient way to setup the script for restarts and simultaneous jobs. A common
setup is to train in one process, and evaluate in a separate process, using the
same timestamp for each process.
"""
import datetime
import functools
import os

from absl import app
from absl import flags
from absl import logging
import edward2 as ed
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import bayesian_rnn_flags
import bayesian_rnn_model

# TODO(dusenberrymw): Expose open-source versions of these files.
import input_fn
import util
import prediction_task
import constants

flags.adopt_module_key_flags(bayesian_rnn_flags)
FLAGS = flags.FLAGS


class Dataset(object):
  """Represents a dataset with train/eval/test/etc. splits.

  Splits are loaded lazily as they are accessed. Consider this a POC of what a
  "Datasets" library could look like (although in the real case, we would
  extend TensorFlow Datasets).
  """

  def __init__(self, task, context_features, sequence_features, batch_size,
               eval_batch_size, feature_engineering_fn, seed):
    """Stores all info needed to generate the dataset."""
    self._train_dataset = None
    self._train_eval_dataset = None
    self._val_dataset = None
    self._test_dataset = None
    self.batch_size = batch_size
    self.eval_batch_size = eval_batch_size
    self._create_dataset = functools.partial(
        input_fn.create_dataset,
        context_features=context_features,
        sequence_features=sequence_features,
        task=task,
        buffer_size=256,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        feature_engineering_fn=feature_engineering_fn,
        seed=seed)

  @property
  def train_dataset(self):
    """Training dataset split for use during training."""
    if self._train_dataset is None:
      self._train_dataset = self._create_dataset(
          mode=tf.estimator.ModeKeys.TRAIN,
          dataset_type=constants.DatasetType.TRAIN,
          batch_size=self.batch_size,
          shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)
    return self._train_dataset

  @property
  def train_eval_dataset(self):
    """Training dataset split for use during evaluation with no shuffling."""
    if self._train_eval_dataset is None:
      self._train_eval_dataset = self._create_dataset(
          mode=tf.estimator.ModeKeys.EVAL,
          dataset_type=constants.DatasetType.TRAIN,
          batch_size=self.eval_batch_size,
          shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)
    return self._train_eval_dataset

  @property
  def val_dataset(self):
    """Validation dataset split."""
    if self._val_dataset is None:
      self._val_dataset = self._create_dataset(
          mode=tf.estimator.ModeKeys.EVAL,
          dataset_type=constants.DatasetType.VALIDATION,
          batch_size=self.eval_batch_size,
          shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)
    return self._val_dataset

  @property
  def test_dataset(self):
    """Testing dataset split."""
    if self._test_dataset is None:
      self._test_dataset = self._create_dataset(
          mode=tf.estimator.ModeKeys.EVAL,
          dataset_type=constants.DatasetType.TEST,
          batch_size=self.eval_batch_size,
          shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)
    return self._test_dataset


def evaluate(model, dataset, task, global_step, num_ece_bins, num_samples,
             ensemble_size, name, steps=-1):
  """Evaluates the model on a dataset."""
  nll_metric = tf.keras.metrics.Mean()
  ece_metric = ed.metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
  aucpr_metric = tf.keras.metrics.AUC(curve="PR")
  aucroc_metric = tf.keras.metrics.AUC(curve="ROC")
  acc_metric = tf.keras.metrics.Accuracy()
  top_k = 5 if task.logits_dimension >= 5 else None
  sensitivity_metric = tf.keras.metrics.Recall(top_k=top_k)
  ppv_metric = tf.keras.metrics.Precision(top_k=top_k)

  for inputs, labels in dataset.take(steps):
    logits = tf.reshape(
        [model(inputs, training=False) for _ in range(num_samples)],
        [num_samples, ensemble_size, -1, task.logits_dimension])
    if task.logits_dimension == 1:
      label_dist = tfp.distributions.Bernoulli(logits)
      labels_1d = labels
      probs = tf.sigmoid(logits)
    else:
      label_dist = tfp.distributions.Multinomial(1, logits)
      labels_1d = tf.math.argmax(labels, axis=-1)
      probs = tf.nn.softmax(logits)
    nll = tf.reduce_mean(
        -tf.reduce_logsumexp(label_dist.log_prob(labels), axis=[0, 1]) +
        tf.math.log(float(num_samples * ensemble_size)))
    probs = tf.math.reduce_mean(probs, axis=[0, 1])  # marginalize
    if task.logits_dimension == 1:
      preds = tf.cast(probs >= 0.5, labels.dtype)
    else:
      preds = tf.math.argmax(probs, axis=-1)
    nll_metric(nll)
    ece_metric(labels_1d, probs)
    aucpr_metric(labels, probs)
    aucroc_metric(labels, probs)
    sensitivity_metric(labels, probs)
    ppv_metric(labels, probs)
    acc_metric(labels_1d, preds)

  nll = nll_metric.result()
  kl = sum(model.losses)
  loss = nll + kl / acc_metric.count
  ece = ece_metric.result()
  aucpr = aucpr_metric.result()
  aucroc = aucroc_metric.result()
  sens = sensitivity_metric.result()
  ppv = ppv_metric.result()
  f1 = (2 * ppv * sens) / (ppv + sens)
  acc = acc_metric.result()
  top_k_suffix = f"@{top_k}" if top_k is not None else ""
  metrics = {"loss": loss, "nll": nll, "kl": kl, "ece": ece, "aucpr": aucpr,
             "aucroc": aucroc, f"sensitivity{top_k_suffix}": sens,
             f"ppv{top_k_suffix}": ppv, f"f1{top_k_suffix}": f1,
             "accuracy": acc}
  loss_metric_names = ["loss", "nll", "kl"]
  other_metric_names = sorted(set(metrics) - set(loss_metric_names))
  logging_string = f"(Eval {name} @ step {int(global_step)})"
  for k in loss_metric_names + other_metric_names:  # preferred order
    logging_string += f", {k}: {metrics[k]:.4f}"
  logging.info(logging_string)
  for k in loss_metric_names:
    tf.summary.scalar(f"loss/{k}", metrics[k], step=global_step)
  for k in other_metric_names:
    tf.summary.scalar(f"metrics/{k}", metrics[k], step=global_step)
  return metrics


def predict(model, dataset, num_samples, ensemble_size, steps=-1):
  """Makes logit predictions with the model."""
  all_logits = []
  all_labels = []
  for inputs, labels in dataset.take(steps):
    logits = [model(inputs, training=False) for _ in range(num_samples)]
    # Reshape from [num_samples, batch_size*ensemble_size, logits_dim] to
    # [ensemble_size, num_samples, batch_size, logits_dim].
    logits = tf.split(logits, num_or_size_splits=ensemble_size, axis=1)
    all_logits.append(logits)
    if labels is not None:
      all_labels.append(labels)
  # Yield logits of shape [ensemble_size, num_samples, N, logits_dim].
  all_logits = tf.concat(all_logits, 2)
  all_labels = tf.concat(all_labels, 0) if all_labels else None
  return all_logits, all_labels


def get_device_and_data_format():
  device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
  if tf.test.is_gpu_available():
    data_format = "channels_first"
  else:
    data_format = "channels_last"
  return device, data_format


class MetricReporter(object):
  """Class that can report metrics to an experiment tracker."""

  def __init__(self, prefix=""):
    """Initializes a metric reporter.

    Args:
      prefix: A string to prepend to each metric name.
    """
    try:
      # TODO(dusenberrymw): Open-source an experiment tracker.
      self.work_unit = get_work_unit_for_tracker()
    except RuntimeError:  # local runs don't use metric reporting
      self.work_unit = None
    self.measurements = {}
    self.prefix = prefix
    self.built = False

  def _build(self, metrics):
    for k in metrics.keys():
      name = f"{self.prefix}_{k}"
      self.measurements[name] = self.work_unit.get_measurement_series(
          label=name)
    self.built = True

  def maybe_report_metrics(self, metrics, global_step):
    """Reports metrics if this job was started remotely.

    Args:
      metrics: A dictionary mapping string names to metric values.
      global_step: An integer global step value.
    """
    if self.work_unit is not None:
      if not self.built:
        self._build(metrics)
      for k, v in metrics.items():
        name = f"{self.prefix}_{k}"
        self.measurements[name].create_measurement(v, step=int(global_step))


class BestExporter(object):
  """Best model exporter."""

  def __init__(self, checkpoint, directory, goal, filename="ckpt"):
    """Set up a best model exporter.

    Args:
      checkpoint: A tf.train.Checkpoint object.
      directory: The path to a directory in which to write checkpoints. This
        object will also save a special file with the name "checkpoint_state" to
        maintain the best metric seen so far.
      goal: The metric optimization goal. Either "max" or "min".
      filename: A string prefix for saved checkpoint filenames.
    """
    self.checkpoint = checkpoint
    if goal == "max":
      self.compare_fn = lambda a, b: a >= b
    else:
      self.compare_fn = lambda a, b: a <= b
    if not tf.io.gfile.exists(directory):
      tf.io.gfile.makedirs(directory)
    self.checkpoint_path = os.path.join(directory, filename)
    self.state_path = os.path.join(directory, "checkpoint_state")
    try:
      self.best_metric = tf.Variable(0., name="best_metric")
      self.state_checkpoint = tf.train.Checkpoint(best_metric=self.best_metric)
      self.state_checkpoint.restore(self.state_path)
    except tf.errors.NotFoundError:
      self.best_metric = None
      self.state_checkpoint = None

  def maybe_export_model(self, current_metric):
    """Exports the model if it is the best seen thus far.

    Args:
      current_metric: A scalar metric value.
    """
    if self.best_metric is None:
      self.best_metric = tf.Variable(current_metric, name="best_metric")
      self.state_checkpoint = tf.train.Checkpoint(best_metric=self.best_metric)
    if self.compare_fn(current_metric, self.best_metric.numpy()):
      self.best_metric.assign(current_metric)
      self.state_checkpoint.write(self.state_path)
      self.checkpoint.write(self.checkpoint_path)


def convert_flags_list_to_string(flags_list):
  """Converts a list of flags into a string with --name=value per line."""
  # Based on the absl.flags.FlagValues.flags_into_string method.
  s = ""
  sorted_flags = sorted(flags_list, key=lambda f: f.name)
  for flag in sorted_flags:
    if flag.value is not None:
      s += flag.serialize() + "\n"
  return s


def setup_and_save_flags():
  """Sets defaults for placeholders and saves the resulting flags."""
  timestamp = datetime.datetime.strftime(datetime.datetime.today(),
                                         "%y%m%d_%H%M%S")
  FLAGS.timestamp = FLAGS.timestamp.format(timestamp=timestamp)
  FLAGS.logdir = FLAGS.logdir.format(timestamp=FLAGS.timestamp)
  FLAGS.model_dir = FLAGS.model_dir.format(timestamp=FLAGS.timestamp)
  FLAGS.predict_dir = FLAGS.predict_dir.format(timestamp=FLAGS.timestamp)
  if not tf.io.gfile.exists(FLAGS.model_dir):
    tf.io.gfile.makedirs(FLAGS.model_dir)
  # NOTE: This saved flags file can be reused in the future via
  # --flagfile=path/to/flags.cfg. Any overrides must be listed after this flag.
  flags_string = convert_flags_list_to_string(
      FLAGS.get_key_flags_for_module(bayesian_rnn_flags))
  flag_file = os.path.join(FLAGS.model_dir, "flags.cfg")
  if (FLAGS.job in ("train", "train_and_eval") and
      not tf.io.gfile.exists(flag_file)):
    with tf.io.gfile.GFile(flag_file, "w") as f:
      f.write(flags_string)


# TODO(dusenberrymw): Shorten this function.
def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  setup_and_save_flags()
  tf.enable_v2_behavior()
  tf.random.set_seed(FLAGS.seed)
  device, _ = get_device_and_data_format()
  logging.info(f"{FLAGS.logdir}, {FLAGS.model_dir}, {FLAGS.predict_dir}, "
               f"{device}")

  # TODO(dusenberrymw): Expose open-source versions of these functions.
  task = prediction_task.get_prediction_task(FLAGS.prediction_task)
  feature_engineering_fn = create_feature_engineering_fn(None)

  logging.info("Loading data...")
  if FLAGS.model == "rank1_bayesian_rnn":
    ensemble_size = FLAGS.ensemble_size
  else:
    ensemble_size = 1
  batch_size = FLAGS.batch_size // ensemble_size
  if FLAGS.eval_batch_size:
    eval_batch_size = FLAGS.eval_batch_size // ensemble_size
  else:
    eval_batch_size = batch_size

  num_train_examples = util.get_num_examples(
      task.data_file_pattern(constants.DatasetType.TRAIN))
  if FLAGS.eval_train_steps == -1:
    eval_train_steps = num_train_examples // eval_batch_size
  else:
    eval_train_steps = FLAGS.eval_train_steps

  def preprocess(inputs):
    """Preprocesses examples."""
    inputs = feature_engineering_fn(inputs)
    try:
      labels = task.extract_labels(inputs)[task.label_key]
    except ValueError:
      labels = None
    if ensemble_size > 1:
      # Rank-1 models require the inputs to be tiled along the batch dimension.
      new_inputs = {}
      for k, v in inputs.items():
        if isinstance(v, tf.SparseTensor):
          new_inputs[k] = tf.sparse.concat(
              sp_inputs=[v]*ensemble_size, axis=0)
        else:
          new_inputs[k] = tf.concat([v]*ensemble_size, axis=0)
      inputs = new_inputs
    return inputs, labels

  dataset = Dataset(
      task=task,
      context_features=FLAGS.context_features,
      sequence_features=FLAGS.sequence_features,
      batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      feature_engineering_fn=preprocess,
      seed=FLAGS.seed)

  if FLAGS.model == "bayesian_rnn":
    model = bayesian_rnn_model.BayesianRNNWithEmbeddings(
        embedding_config=task.embedding_config(),
        sequence_features=FLAGS.sequence_features,
        context_features=FLAGS.context_features,
        rnn_dim=FLAGS.rnn_dim,
        num_rnn_layers=FLAGS.num_rnn_layers,
        hidden_layer_dim=FLAGS.hidden_layer_dim,
        output_layer_dim=task.logits_dimension,
        rnn_uncertainty=FLAGS.uncertainty_rnn,
        hidden_uncertainty=FLAGS.uncertainty_hidden,
        output_uncertainty=FLAGS.uncertainty_output,
        bias_uncertainty=FLAGS.uncertainty_biases,
        embeddings_uncertainty=FLAGS.uncertainty_embeddings,
        prior_stddev=FLAGS.prior_stddev,
        l2=float(FLAGS.l2),
        clip_norm=FLAGS.clip_norm,
        bagging_time_precision=FLAGS.bagging_time_precision,
        bagging_aggregate_older_than=FLAGS.bagging_aggregate_older_than,
        embedding_dimension_multiplier=FLAGS.embedding_dimension_multiplier,
        dense_feature_name=FLAGS.dense_feature_name,
        dense_feature_value=FLAGS.dense_feature_value,
        dense_feature_unit=FLAGS.dense_feature_unit,
        dense_embedding_dimension=FLAGS.dense_embedding_dimension,
        top_n_dense=FLAGS.top_n_dense,
        num_ids_per_dense_feature=FLAGS.num_ids_per_dense_feature,
        dense_stats_config_path=FLAGS.stats_config_path)
  else:  # rank1_bayesian_rnn
    model = bayesian_rnn_model.Rank1BayesianRNNWithEmbeddings(
        embedding_config=task.embedding_config(),
        sequence_features=FLAGS.sequence_features,
        context_features=FLAGS.context_features,
        rnn_dim=FLAGS.rnn_dim,
        num_rnn_layers=FLAGS.num_rnn_layers,
        hidden_layer_dim=FLAGS.hidden_layer_dim,
        output_layer_dim=task.logits_dimension,
        embeddings_initializer=FLAGS.embeddings_initializer,
        embeddings_regularizer=FLAGS.embeddings_regularizer,
        alpha_initializer=FLAGS.alpha_initializer,
        gamma_initializer=FLAGS.gamma_initializer,
        alpha_regularizer=FLAGS.alpha_regularizer,
        gamma_regularizer=FLAGS.gamma_regularizer,
        use_additive_perturbation=FLAGS.use_additive_perturbation,
        ensemble_size=ensemble_size,
        random_sign_init=float(FLAGS.random_sign_init),
        dropout_rate=float(FLAGS.dropout_rate),
        prior_mean=float(FLAGS.prior_mean),
        prior_stddev=float(FLAGS.prior_stddev),
        l2=float(FLAGS.l2),
        clip_norm=float(FLAGS.clip_norm),
        bagging_time_precision=FLAGS.bagging_time_precision,
        bagging_aggregate_older_than=FLAGS.bagging_aggregate_older_than,
        embedding_dimension_multiplier=FLAGS.embedding_dimension_multiplier,
        dense_feature_name=FLAGS.dense_feature_name,
        dense_feature_value=FLAGS.dense_feature_value,
        dense_feature_unit=FLAGS.dense_feature_unit,
        dense_embedding_dimension=FLAGS.dense_embedding_dimension,
        top_n_dense=FLAGS.top_n_dense,
        num_ids_per_dense_feature=FLAGS.num_ids_per_dense_feature,
        dense_stats_config_path=FLAGS.stats_config_path)

  # TODO(dusenberrymw): Autograph currently can only be used on the inner
  # `model.bayesian_rnn` and only with `experimental_relax_shapes=True` enabled.
  # Even with this, it results in a memory leak leading to OOM errors.
  # model.bayesian_rnn.call = tf.function(model.bayesian_rnn.call,
  #                                       experimental_relax_shapes=True)
  global_step = tf.Variable(0, name="global_step", dtype=tf.int64)
  learning_rate_fn = lambda x: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(learning_rate_fn(global_step))

  checkpoint = tf.train.Checkpoint(
      model=model, global_step=global_step, optimizer=optimizer)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=FLAGS.model_dir,
      max_to_keep=FLAGS.max_to_keep if FLAGS.max_to_keep > 0 else None)
  if checkpoint_manager.latest_checkpoint:
    logging.info("Loading existing model...")
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  elif FLAGS.job in ("train", "train_and_eval"):
    checkpoint_manager.save()

  train_reporter = MetricReporter("train")
  val_reporter = MetricReporter("val")
  test_reporter = MetricReporter("test")
  exporter = BestExporter(checkpoint, os.path.join(FLAGS.model_dir, "best"),
                          FLAGS.metric_goal)

  def eval_dataset(dataset, name, steps=-1):
    """Evaluates the model on a dataset."""
    writer = tf.summary.create_file_writer(f"{FLAGS.logdir}/{name}_eval")
    with tf.device(device), writer.as_default():
      metrics = evaluate(
          model, dataset=dataset, task=task, global_step=global_step,
          num_ece_bins=FLAGS.num_ece_bins, num_samples=FLAGS.num_eval_samples,
          ensemble_size=ensemble_size, name=name, steps=steps)
    writer.flush()
    return metrics

  if FLAGS.job in ("train", "train_and_eval"):
    # TODO(dusenberrymw): Consider converting to a function outside of main.
    logging.info("Training...")

    def eval_and_export():
      train_metrics = eval_dataset(
          dataset.train_eval_dataset, "train", eval_train_steps)
      train_reporter.maybe_report_metrics(train_metrics, global_step)
      val_metrics = eval_dataset(dataset.val_dataset, "val")
      val_reporter.maybe_report_metrics(val_metrics, global_step)
      exporter.maybe_export_model(val_metrics[FLAGS.metric])
      if FLAGS.eval_test_in_loop:
        test_metrics = eval_dataset(dataset.test_dataset, "test")
        test_reporter.maybe_report_metrics(test_metrics, global_step)

    train_writer = tf.summary.create_file_writer(f"{FLAGS.logdir}/train")
    with tf.device(device), train_writer.as_default():
      for inputs, labels in (dataset.train_dataset.skip(
          global_step).take(FLAGS.max_steps - global_step)):
        with tf.GradientTape() as tape:
          logits = tf.reshape(
              [model(inputs) for _ in range(FLAGS.num_train_samples)],
              [FLAGS.num_train_samples, ensemble_size, -1,
               task.logits_dimension])
          if task.logits_dimension == 1:
            label_dist = tfp.distributions.Bernoulli(logits)
          else:
            label_dist = tfp.distributions.Multinomial(1, logits)

          if FLAGS.nll == "mixture":
            nll = tf.reduce_mean(
                -tf.reduce_logsumexp(label_dist.log_prob(labels), axis=[0, 1]) +
                tf.math.log(float(FLAGS.num_train_samples * ensemble_size)))
          else:  # average NLL
            nll = -tf.reduce_mean(label_dist.log_prob(labels))

          kl = sum(model.losses)
          kl = kl / num_train_examples * FLAGS.kl_scale * tf.minimum(
              1.,
              tf.cast(global_step + 1, tf.float32) / FLAGS.kl_annealing_steps)
          loss = nll + kl

        if int(global_step) % FLAGS.log_steps == 0:
          checkpoint_manager.save()
          tf.summary.scalar("loss/nll", nll, step=global_step)
          tf.summary.scalar("loss/kl", kl, step=global_step)
          tf.summary.scalar("loss/loss", loss, step=global_step)
          tf.summary.scalar(
              "loss/learning_rate", optimizer.learning_rate, step=global_step)
          logging.info(f"(Train @ step {int(global_step)}/{FLAGS.max_steps}) "
                       f"loss: {loss:.4f}, nll: {nll:.4f}, kl: {kl:.4f}")
          if FLAGS.job == "train_and_eval":
            eval_and_export()
          train_writer.flush()

        grads = tape.gradient(loss, model.trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads, FLAGS.clip_norm)
        if int(global_step) % FLAGS.log_steps == 0:
          tf.summary.scalar(
              "loss/grads/global_norm", global_norm, step=global_step)
        # Separate learning rate implementation.
        if (FLAGS.model == "rank1_bayesian_rnn" and
            FLAGS.fast_weight_lr_multiplier != 1.0):
          grads_and_vars = []
          for grad, var in zip(grads, model.trainable_variables):
            # Apply different learning rate on the fast weight approximate
            # posterior/prior parameters.
            if ("kernel" not in var.name and
                "recurrent_kernel" not in var.name and
                "bias" not in var.name):
              grads_and_vars.append(
                  (grad * FLAGS.fast_weight_lr_multiplier, var))
            else:
              grads_and_vars.append((grad, var))
        else:
          grads_and_vars = list(zip(grads, model.trainable_variables))
        optimizer.learning_rate = learning_rate_fn(global_step)
        optimizer.apply_gradients(grads_and_vars)
        global_step.assign_add(1)
        del grads, grads_and_vars, logits  # reduces memory pressure

    checkpoint_manager.save()
    if FLAGS.job == "train_and_eval":
      # Final step evaluation.
      eval_and_export()

      # Best checkpoint evaluation.
      checkpoint.restore(os.path.join(FLAGS.model_dir, "best", "ckpt"))
      train_metrics = eval_dataset(
          dataset.train_eval_dataset, "train-best", eval_train_steps)
      MetricReporter("train-best").maybe_report_metrics(train_metrics,
                                                        FLAGS.max_steps)
      val_metrics = eval_dataset(dataset.val_dataset, "val-best")
      MetricReporter("val-best").maybe_report_metrics(val_metrics,
                                                      FLAGS.max_steps)
      if FLAGS.eval_test_in_loop:
        test_metrics = eval_dataset(dataset.test_dataset, "test-best")
        MetricReporter("test-best").maybe_report_metrics(test_metrics,
                                                         FLAGS.max_steps)

  elif (FLAGS.job in ("eval_train", "eval_val") or
        (FLAGS.job == "eval_test" and FLAGS.eval_test_in_loop)):
    # TODO(dusenberrymw): Consider converting to a function outside of main.
    logging.info("Evaluating...")
    prev_global_step = -1
    while global_step <= FLAGS.max_steps:
      try:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS.model_dir))
      except tf.errors.NotFoundError:
        continue
      if int(global_step) > prev_global_step:
        if FLAGS.job == "eval_train":
          metrics = eval_dataset(dataset.train_eval_dataset, "train",
                                 eval_train_steps)
          train_reporter.maybe_report_metrics(metrics, global_step)
        elif FLAGS.job == "eval_val":
          metrics = eval_dataset(dataset.val_dataset, "val")
          val_reporter.maybe_report_metrics(metrics, global_step)
          exporter.maybe_export_model(metrics[FLAGS.metric])
        else:  # "eval_test"
          metrics = eval_dataset(dataset.test_dataset, "test")
          test_reporter.maybe_report_metrics(metrics, global_step)
        prev_global_step = int(global_step)
      if global_step == FLAGS.max_steps:
        break

    # Final best checkpoint evaluation.
    checkpoint.restore(os.path.join(FLAGS.model_dir, "best", "ckpt"))
    if FLAGS.job == "eval_train":
      metrics = eval_dataset(dataset.train_eval_dataset, "train-best",
                             eval_train_steps)
      MetricReporter("train-best").maybe_report_metrics(metrics,
                                                        FLAGS.max_steps)
    elif FLAGS.job == "eval_val":
      metrics = eval_dataset(dataset.val_dataset, "val-best")
      MetricReporter("val-best").maybe_report_metrics(metrics, FLAGS.max_steps)
    else:  # "eval_test"
      metrics = eval_dataset(dataset.test_dataset, "test-best")
      MetricReporter("test-best").maybe_report_metrics(metrics, FLAGS.max_steps)

  elif FLAGS.job in ("predict_train", "predict_val", "predict_test"):
    # TODO(dusenberrymw): Consider converting to a function outside of main.
    logging.info("Predicting...")
    checkpoint.restore(os.path.join(FLAGS.model_dir, "best", "ckpt"))
    pred_fn = functools.partial(
        predict, model=model, num_samples=FLAGS.num_eval_samples,
        ensemble_size=ensemble_size)
    with tf.device(device):
      if FLAGS.job == "predict_train":
        all_logits, all_labels = pred_fn(
            dataset=dataset.train_eval_dataset)
        save_dir = os.path.join(FLAGS.predict_dir, "train")
      elif FLAGS.job == "predict_val":
        all_logits, all_labels = pred_fn(dataset=dataset.val_dataset)
        save_dir = os.path.join(FLAGS.predict_dir, "val")
      else:  # "predict_test"
        all_logits, all_labels = pred_fn(dataset=dataset.test_dataset)
        save_dir = os.path.join(FLAGS.predict_dir, "test")
    if not tf.io.gfile.exists(save_dir):
      tf.io.gfile.makedirs(save_dir)
    with tf.io.gfile.GFile(os.path.join(save_dir, "global_step.txt"), "w") as f:
      f.write(f"{int(global_step)}")
    with tf.io.gfile.GFile(os.path.join(save_dir, "logits.npy"), "w") as f:
      np.save(f, all_logits)
    if all_labels is not None:
      with tf.io.gfile.GFile(os.path.join(save_dir, "labels.npy"), "w") as f:
        np.save(f, all_labels)

  else:  # "eval_test"
    logging.info("Evaluating...")
    eval_dataset(dataset.test_dataset, "test")


if __name__ == "__main__":
  app.run(main)
