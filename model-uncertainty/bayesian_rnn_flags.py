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

"""Flags for the Bayesian RNN train/eval/predict script."""
from absl import flags

# NOTE: The following flags are specific to the Bayesian RNN model.
flags.DEFINE_integer(
    "bagging_time_precision",
    default=86400,
    help=("Precision for bagging in seconds. For example, 86400 represents "
          "day-level bagging."))
flags.DEFINE_integer(
    "bagging_aggregate_older_than",
    default=-1,
    help=("Events older than this (in seconds) will all be aggregated into the "
          "same bag. `-1` indicates no aggregation."))
flags.DEFINE_list(
    "context_features",
    default=[
        "sequenceLength", "Patient.birthDate", "timestamp", "Patient.gender"
    ],
    help="List of the context (per-sequence) features to process.")
flags.DEFINE_float(
    "embedding_dimension_multiplier",
    default=0.85827,
    help="Multiplier for the embedding dimensionality.")
flags.DEFINE_float(
    "dense_embedding_dimension",
    default=32,
    help="Dense embedding dimensionality.")
flags.DEFINE_string(
    "dense_feature_name",
    default="Observation.code",
    help=("A feature in sequence_features representing the names of all "
          "continuous inputs. Used for dense embedding."))
flags.DEFINE_string(
    "dense_feature_value",
    default="Observation.value.quantity.value",
    help=("A feature in sequence_features representing the values of all "
          "continuous inputs. Used for dense embedding."))
flags.DEFINE_string(
    "dense_feature_unit",
    default="Observation.value.quantity.unit",
    help=("A feature in sequence_features representing the units of all "
          "continuous input values. Used for dense embedding."))
flags.DEFINE_integer(
    "hidden_layer_dim", default=128, help="Hidden layer output dimensionality.")
flags.DEFINE_enum(
    "model",
    default="bayesian_rnn",
    enum_values=["bayesian_rnn", "rank1_bayesian_rnn"],
    help="Which model to use.")
flags.DEFINE_integer(
    "num_rnn_layers", default=1, help="Number of stacked RNN cells.")
flags.DEFINE_integer(
    "num_ids_per_dense_feature",
    default=5,
    help="Number of embedding IDs per dense feature.")
flags.DEFINE_float(
    "prior_stddev",
    default=1.,
    help="The standard deviation for the Normal prior.")
flags.DEFINE_integer(
    "rnn_dim", default=256, help="RNN cell output dimensionality.")
flags.DEFINE_list(
    "sequence_features",
    default=[
        "Composition.section.text.div.tokenized", "deltaTime",
        "Observation.code", "Observation.value.quantity.value",
        "Observation.value.quantity.unit"
    ],
    help="List of the context (per-sequence) features to process.")
flags.DEFINE_integer(
    "top_n_dense",
    default=100,
    help=("Number of top dense features (by frequency) to retain for model "
          "input."))
flags.DEFINE_boolean(
    "uncertainty_biases",
    default=False,
    help="Whether or not to use Bayesian bias parameters.")
flags.DEFINE_boolean(
    "uncertainty_embeddings",
    default=False,
    help="Whether or not to use a Bayesian embedding layer.")
flags.DEFINE_boolean(
    "uncertainty_hidden",
    default=False,
    help="Whether or not to use a Bayesian hidden dense layer.")
flags.DEFINE_boolean(
    "uncertainty_output",
    default=False,
    help="Whether or not to use a Bayesian output dense layer.")
flags.DEFINE_boolean(
    "uncertainty_rnn",
    default=False,
    help="Whether or not to use a Bayesian RNN layer.")

# NOTE: The following flags are specific to the rank-1 Bayesian RNN.
flags.DEFINE_string("embeddings_initializer", "trainable_normal",
                    "Initializer name for the embedding parameters.")
flags.DEFINE_string("embeddings_regularizer", "normal_kl_divergence",
                    "Regularizer name for the embedding parameters.")
flags.DEFINE_string("alpha_initializer", "trainable_normal",
                    "Initializer name for the alpha parameters.")
flags.DEFINE_string("alpha_regularizer", "normal_kl_divergence",
                    "Regularizer name for the alpha parameters.")
flags.DEFINE_string("gamma_initializer", "trainable_normal",
                    "Initializer name for the gamma parameters.")
flags.DEFINE_string("gamma_regularizer", "normal_kl_divergence",
                    "Regularizer name for the gamma parameters.")
flags.DEFINE_boolean("use_additive_perturbation", False,
                     "Use additive perturbations instead of multiplicative.")
flags.DEFINE_float("dropout_rate", 1e-3,
                   "Dropout rate. Only used if alpha/gamma initializers are, "
                   "e.g., trainable normal.")
flags.DEFINE_float("prior_mean", 1., "Prior mean.")
flags.DEFINE_integer("ensemble_size", 4, "Size of ensemble.")
flags.DEFINE_float("random_sign_init", 0.5,
                   "Use random sign init for fast weights.")
flags.DEFINE_float("fast_weight_lr_multiplier", 1.0,
                   "fast weights lr multiplier.")
flags.DEFINE_float("l2", 1e-4, "L2 coefficient.")

# NOTE: The following flags are specific to the train/eval/predict loops.
flags.DEFINE_integer(
    "batch_size", default=64, help="Batch size during training.")
flags.DEFINE_float(
    "clip_norm",
    default=7.29199,
    help="Threshold for global norm gradient clipping.")
flags.DEFINE_integer(
    "eval_batch_size",
    default=None,
    help="Batch size during evaluation. Defaults to `batch_size`.")
flags.DEFINE_boolean(
    "eval_test_in_loop",
    default=False,
    help="Whether or not to evaluate on the test set in a loop. [dangerous]")
flags.DEFINE_integer(
    "eval_train_steps",
    default=-1,
    help=("Number of training batches over which to evaluate. Useful if "
          "evaluation is too costly. Default is to evaluate over the full "
          "dataset."))
flags.DEFINE_enum(
    "job",
    default="train_and_eval",
    enum_values=[
        "train_and_eval", "train_and_eval_parallel", "train", "eval_train",
        "eval_val", "eval_test", "predict_train", "predict_val", "predict_test"
    ],
    help="Which job(s) to run.")
flags.DEFINE_integer(
    "kl_annealing_steps",
    default=5000,
    help="Number of steps over which to anneal the KL term to 1.")
flags.DEFINE_float(
    "kl_scale", default=1., help="Scaling factor for the KL term.")
flags.DEFINE_float(
    "learning_rate", default=0.00030352, help="Learning rate during training.")
flags.DEFINE_string(
    "logdir",  # `log_dir` is already defined by absl
    default="/tmp/medical_uncertainty/bayesian_rnn/logs/{timestamp}",
    help="Directory in which to write TensorBoard logs.")
flags.DEFINE_integer(
    "log_steps",
    default=100,
    help="Frequency, in steps, of TensorBoard logging.")
flags.DEFINE_integer(
    "max_steps", default=10000, help="Number of steps over which to train.")
flags.DEFINE_integer(
    "max_to_keep",
    default=20,
    help="Number of checkpoints to keep. If -1, all checkpoints are kept.")
flags.DEFINE_string(
    "metric",
    default="loss",
    help=("The metric to optimize with Vizier and to use for best model "
          "exporting. Valid metric names are those reported to XManager, such "
          "as 'loss', 'nll', 'kl', 'ece', 'aucpr', 'aucroc', and 'accuracy'"))
flags.DEFINE_enum(
    "metric_goal",
    default="min",
    enum_values=["min", "max"],
    help="The optimization goal for Vizier and best model exporting.")
flags.DEFINE_string(
    "model_dir",
    default="/tmp/medical_uncertainty/bayesian_rnn/models/{timestamp}",
    help="Directory in which to save model checkpoints.")
flags.DEFINE_enum(
    "nll",
    default="average",
    enum_values=["average", "mixture"],
    help=("The negative log likelihood formulation to use given a mixture "
          "distribution over the model parameters, where 'average' corresponds "
          "to computing the average per-component NLL, and 'mixture' "
          "corresponds to the negative log marginal likelihood."))
flags.DEFINE_integer(
    "num_ece_bins", default=15, help="Number of bins for the ECE metric.")
flags.DEFINE_integer(
    "num_eval_samples",
    default=1,
    help="Number of model predictions to sample per example at eval time.")
flags.DEFINE_integer(
    "num_train_samples",
    default=1,
    help="Number of model predictions to sample per example at train time.")
flags.DEFINE_string(
    "predict_dir",
    default="/tmp/medical_uncertainty/bayesian_rnn/predictions/{timestamp}",
    help="Directory in which to save model predictions.")
flags.DEFINE_string(
    "prediction_task", "mortality",
    "Prediction task. Use either existing tasks, `custom` for "
    "new custom prediction tasks, or `custom:base_task_name` "
    "such as `custom:mortality` for overriding existing tasks "
    "with custom params.")
flags.DEFINE_integer("seed", default=0, help="Random seed.")
# TODO(dusenberrymw): Expose an open-source version of this file.
flags.DEFINE_string(
    "stats_config_path",
    default="/path/to/tf-dense-input.config",
    help="Path to dataset-specific dense config stats.")
flags.DEFINE_string(
    "timestamp",
    default="{timestamp}",
    help="Timestamp to use. Default is current date-time.")
