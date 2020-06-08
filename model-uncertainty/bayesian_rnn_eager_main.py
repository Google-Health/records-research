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

"""Bayesian RNN Eager train/eval/predict script.

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
import random

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
               buffer_size, shuffle, num_parallel_calls,
               feature_engineering_fn, seed=None):
    """Stores all info needed to generate the dataset."""
    self._train_dataset = None
    self._train_eval_dataset = None
    self._val_dataset = None
    self._test_dataset = None
    self._create_dataset = functools.partial(
        input_fn.create_dataset,
        context_features=context_features,
        sequence_features=sequence_features,
        task=task,
        batch_size=batch_size,
        buffer_size=buffer_size,
        shuffle=shuffle,
        num_parallel_calls=num_parallel_calls,
        feature_engineering_fn=feature_engineering_fn,
        seed=seed)

  @property
  def train_dataset(self):
    """Training dataset split for use during training."""
    if self._train_dataset is None:
      with tf.device("/cpu:0"):
        self._train_dataset = self._create_dataset(
            mode=tf.estimator.ModeKeys.TRAIN,
            dataset_type=constants.DatasetType.TRAIN)
    return self._train_dataset

  @property
  def train_eval_dataset(self):
    """Training dataset split for use during evaluation with no shuffling."""
    if self._train_eval_dataset is None:
      with tf.device("/cpu:0"):
        self._train_eval_dataset = self._create_dataset(
            mode=tf.estimator.ModeKeys.EVAL,
            dataset_type=constants.DatasetType.TRAIN,
            shuffle=False)
    return self._train_eval_dataset

  @property
  def val_dataset(self):
    """Validation dataset split."""
    if self._val_dataset is None:
      with tf.device("/cpu:0"):
        self._val_dataset = self._create_dataset(
            mode=tf.estimator.ModeKeys.EVAL,
            dataset_type=constants.DatasetType.VALIDATION,
            shuffle=False)
    return self._val_dataset

  @property
  def test_dataset(self):
    """Testing dataset split."""
    if self._test_dataset is None:
      with tf.device("/cpu:0"):
        self._test_dataset = self._create_dataset(
            mode=tf.estimator.ModeKeys.EVAL,
            dataset_type=constants.DatasetType.TEST,
            shuffle=False)
    return self._test_dataset


def preprocess(examples, augment=False):
  """Preprocesses examples."""
  del augment  # unused arg
  # TODO(dusenberrymw): Attempting to use `task.extract_labels` within this
  # function, which is used within a `dataset.map` call, results in the
  # following warning about resource creation within map statements being
  # unsupported:
  #   UserWarning: Creating resources inside a function passed to Dataset.map()
  #   is not supported. Create each resource outside the function, and capture
  #   it inside the function to use it.
  # Since `tf.data.Dataset.map` automatically invokes TF Autograph to trace and
  # convert the function to a graph, I suspect it is due to added ops that
  # create resources. See tensorflow/python/data/ops/dataset_ops.py,
  # tensorflow/python/training/tracking/tracking.py, and
  # tensorflow/python/data/kernel_tests/map_test.py.
  # inputs, labels = examples, task.extract_labels(examples)[task.label_key]
  # return {"inputs": inputs, "labels": labels}
  return examples


def preprocess_augmented(examples):
  return preprocess(examples, augment=True)


def evaluate(model, dataset, task, global_step, num_ece_bins, num_samples, name,
             steps=-1):
  """Evaluates the model on a dataset."""
  nll_metric = tf.keras.metrics.Mean()
  ece_metric = ed.metrics.ExpectedCalibrationError(
      num_classes=task.logits_dimension if task.logits_dimension > 1 else 2,
      num_bins=num_ece_bins)
  aucpr_metric = tf.keras.metrics.AUC(curve="PR")
  aucroc_metric = tf.keras.metrics.AUC(curve="ROC")
  acc_metric = tf.keras.metrics.Accuracy()
  top_k = 5 if task.logits_dimension >= 5 else None
  sensitivity_metric = tf.keras.metrics.Recall(top_k=top_k)
  ppv_metric = tf.keras.metrics.Precision(top_k=top_k)

  for examples in (dataset.take(steps).map(preprocess).prefetch(
      tf.data.experimental.AUTOTUNE)):
    # TODO(dusenberrymw): Currently, attempting to execute `task.extract_labels`
    # on a GPU results in an error due to not being able to copy a Tensor with
    # type string to the GPU.
    with tf.device("/cpu:0"):
      inputs, labels = examples, task.extract_labels(examples)[task.label_key]
    logits = [model(inputs, training=False) for _ in range(num_samples)]
    logits = tf.math.reduce_mean(logits, axis=0)  # marginalize over parameters
    if logits.shape[-1] == 1:
      label_dist = tfp.distributions.Bernoulli(logits)
      probs = tf.sigmoid(logits)
      preds = tf.cast(probs >= 0.5, labels.dtype)
      labels_1d = labels
    else:
      label_dist = tfp.distributions.Multinomial(1, logits)
      probs = tf.nn.softmax(logits)
      preds = tf.math.argmax(probs, axis=-1)
      labels_1d = tf.math.argmax(labels, axis=-1)
    nll_metric(-label_dist.log_prob(labels))
    ece_metric(labels_1d, probs)
    aucpr_metric(labels, probs)
    aucroc_metric(labels, probs)
    sensitivity_metric(labels, probs)
    ppv_metric(labels, probs)
    acc_metric(labels_1d, preds)

  nll = nll_metric.result()
  kl = sum(model.losses)
  loss = nll + kl / nll_metric.count
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


def predict(model, dataset, task, num_samples=1, steps=-1):
  """Makes logit predictions with the model."""
  all_logits = []
  all_labels = []
  for examples in (dataset.take(steps).map(preprocess).prefetch(
      tf.data.experimental.AUTOTUNE)):
    try:
      # TODO(dusenberrymw): Currently, attempting to execute
      # `task.extract_labels` on a GPU results in an error due to not being able
      # to copy a Tensor with type string to the GPU.
      with tf.device("/cpu:0"):
        labels = task.extract_labels(examples)[task.label_key]
    except ValueError:
      labels = None
    logits = [model(examples, training=False) for _ in range(num_samples)]
    all_logits.append(logits)
    if labels is not None:
      all_labels.append(labels)
  all_logits = tf.concat(all_logits, 1)  # [num_samples, N, logits_dim]
  all_labels = tf.concat(all_labels, 0) if all_labels else None
  return all_logits, all_labels


def get_device_and_data_format():
  device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
  if tf.test.is_gpu_available():
    data_format = "channels_first"
  else:
    data_format = "channels_last"
  return device, data_format


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
  if FLAGS.seed is None:
    FLAGS.seed = random.randint(0, 1e9)
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

  tf.enable_v2_behavior()
  tf.random.set_seed(FLAGS.seed)
  setup_and_save_flags()
  device, _ = get_device_and_data_format()
  logging.info(f"{FLAGS.logdir}, {FLAGS.model_dir}, {FLAGS.predict_dir}, "
               f"{device}")


  # TODO(dusenberrymw): Expose an open-source version of this function.
  task = prediction_task.get_prediction_task(FLAGS.prediction_task)

  logging.info("Loading data...")
  dataset = Dataset(
      task=task,
      context_features=FLAGS.context_features,
      sequence_features=FLAGS.sequence_features,
      batch_size=FLAGS.batch_size,
      buffer_size=FLAGS.buffer_size,
      shuffle=FLAGS.shuffle,
      num_parallel_calls=FLAGS.num_parallel_calls,
      feature_engineering_fn=model_instance.create_feature_engineering_fn(None))

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

  # TODO(dusenberrymw): Using Autograph here currently results in a memory leak.
  # model.call = tf.function(model.call)
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

  def eval_dataset(dataset, log_dir_suffix, name, steps=-1):
    """Evaluates the model on a dataset."""
    writer = tf.summary.create_file_writer(f"{FLAGS.logdir}/{log_dir_suffix}")
    with tf.device(device), writer.as_default():
      metrics = evaluate(
          model, dataset=dataset, task=task, global_step=global_step,
          num_ece_bins=FLAGS.num_ece_bins, num_samples=FLAGS.num_samples,
          name=name, steps=steps)
    writer.flush()
    return metrics

  def eval_train(steps=-1):
    return eval_dataset(dataset=dataset.train_eval_dataset,
                        log_dir_suffix="train_eval", name="train", steps=steps)

  def eval_val():
    return eval_dataset(dataset=dataset.val_dataset,
                        log_dir_suffix="val_eval", name="val")

  def eval_test():
    return eval_dataset(dataset=dataset.test_dataset,
                        log_dir_suffix="test_eval", name="test")

  if FLAGS.job in ("train", "train_and_eval"):
    # TODO(dusenberrymw): Consider converting to a function outside of main.
    logging.info("Training...")
    train_writer = tf.summary.create_file_writer(f"{FLAGS.logdir}/train")
    num_examples = util.get_num_examples(
        task.data_file_pattern(constants.DatasetType.TRAIN))
    with tf.device(device), train_writer.as_default():
      for examples in (dataset.train_dataset.skip(global_step).take(
          FLAGS.max_steps - global_step).map(preprocess_augmented).prefetch(
              tf.data.experimental.AUTOTUNE)):
        # TODO(dusenberrymw): Currently, attempting to execute
        # `task.extract_labels` on a GPU results in an error due to not being
        # able to copy a Tensor with type string to the GPU.
        with tf.device("/cpu:0"):
          labels = task.extract_labels(examples)[task.label_key]
        inputs = examples  # NOTE: task.extract_labels mutates examples.

        with tf.GradientTape() as tape:
          logits = model(inputs)
          if logits.shape[-1] == 1:
            label_dist = tfp.distributions.Bernoulli(logits)
          else:
            label_dist = tfp.distributions.Multinomial(1, logits)
          nll = -tf.reduce_mean(label_dist.log_prob(labels))
          kl = sum(model.losses)
          kl = kl / num_examples * FLAGS.kl_scale * tf.minimum(
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
            eval_train(FLAGS.eval_train_steps)
            eval_val()
          train_writer.flush()

        grads = tape.gradient(loss, model.trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads, FLAGS.clip_norm)
        if int(global_step) % FLAGS.log_steps == 0:
          tf.summary.scalar(
              "loss/grads/global_norm", global_norm, step=global_step)
        grads_and_vars = list(zip(grads, model.trainable_variables))
        optimizer.learning_rate = learning_rate_fn(global_step)
        optimizer.apply_gradients(grads_and_vars)
        global_step.assign_add(1)
        del grads, grads_and_vars, logits  # reduces memory pressure

    checkpoint_manager.save()
    if FLAGS.job == "train_and_eval":
      eval_train(FLAGS.eval_train_steps)
      eval_val()

  elif (FLAGS.job in ("eval_train", "eval_val") or
        (FLAGS.job == "eval_test" and FLAGS.eval_test_in_loop)):
    # TODO(dusenberrymw): Consider converting to a function outside of main.
    logging.info("Evaluating...")
    exporter = BestExporter(checkpoint, os.path.join(FLAGS.model_dir, "best"),
                            FLAGS.metric_goal)
    prev_global_step = -1
    while global_step <= FLAGS.max_steps:
      try:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS.model_dir))
      except tf.errors.NotFoundError:
        continue
      if int(global_step) > prev_global_step:
        if FLAGS.job == "eval_train":
          eval_train(FLAGS.eval_train_steps)
        elif FLAGS.job == "eval_val":
          metrics = eval_val()
          exporter.maybe_export_model(metrics[FLAGS.metric])
        else:  # "eval_test"
          eval_test()  # be careful with this...
        prev_global_step = int(global_step)
      if global_step == FLAGS.max_steps:
        break

  elif FLAGS.job in ("predict_train", "predict_val", "predict_test"):
    # TODO(dusenberrymw): Consider converting to a function outside of main.
    logging.info("Predicting...")
    checkpoint.restore(os.path.join(FLAGS.model_dir, "best", "ckpt"))
    with tf.device(device):
      if FLAGS.job == "predict_train":
        all_logits, all_labels = predict(
            model, dataset.train_eval_dataset, task, FLAGS.num_samples)
        save_dir = os.path.join(FLAGS.predict_dir, "train")
      elif FLAGS.job == "predict_val":
        all_logits, all_labels = predict(
            model, dataset.val_dataset, task, FLAGS.num_samples)
        save_dir = os.path.join(FLAGS.predict_dir, "val")
      else:  # "predict_test"
        all_logits, all_labels = predict(
            model, dataset.test_dataset, task, FLAGS.num_samples)
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
    eval_test()


if __name__ == "__main__":
  app.run(main)
