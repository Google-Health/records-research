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

"""Tests for the Bayesian RNN Eager train/eval/predict script."""
import os
import tempfile

from absl import flags
from absl.testing import flagsaver
import tensorflow.compat.v2 as tf

import bayesian_rnn_eager_main
import bayesian_rnn_flags

# TODO(dusenberrymw): Expose open-source versions of these files.
import test_input
import resources

flags.adopt_module_key_flags(bayesian_rnn_flags)
FLAGS = flags.FLAGS

# TODO(dusenberrymw): Expose open-source versions of these files.
TESTDATA_DIR = ("test/testdata/integration/")


class BayesianRnnEagerMainTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    seqex_list = [
        test_input.read_seqex_ascii("example-0.ascii",
                                    os.path.join(TESTDATA_DIR, "seqex"))
    ]
    input_data_dir = test_input.create_input_data_dir(
        seqex_list, "inpatient_at_24hrs", TESTDATA_DIR, sharded=True)
    FLAGS.logdir = tempfile.mkdtemp()
    FLAGS.model_dir = tempfile.mkdtemp()
    FLAGS.predict_dir = tempfile.mkdtemp()
    FLAGS.input_dir = input_data_dir
    FLAGS.stats_config_path = os.path.join(
        resources.GetRunfilesDir(), TESTDATA_DIR, "VOCAB/dense_stats_config")
    FLAGS.batch_size = 2
    FLAGS.rnn_dim = 32
    FLAGS.log_steps = 2
    FLAGS.max_steps = 5
    FLAGS.eval_train_steps = 2
    FLAGS.uncertainty_embeddings = True
    FLAGS.uncertainty_rnn = True
    FLAGS.uncertainty_hidden = True
    FLAGS.uncertainty_output = True
    FLAGS.uncertainty_biases = True

  @flagsaver.flagsaver
  def testBayesianRNNEagerMain(self):
    bayesian_rnn_eager_main.main([])
    self.assertNotEmpty(os.listdir(FLAGS.logdir))
    self.assertNotEmpty(os.listdir(FLAGS.model_dir))
    self.assertEmpty(os.listdir(FLAGS.predict_dir))

  @flagsaver.flagsaver
  def testBayesianRNNEagerMainTrainEvalPredict(self):
    FLAGS.job = "train"
    bayesian_rnn_eager_main.main([])
    self.assertNotEmpty(os.listdir(FLAGS.logdir))
    self.assertNotEmpty(os.listdir(FLAGS.model_dir))
    self.assertEmpty(os.listdir(FLAGS.predict_dir))

    FLAGS.job = "eval_val"
    FLAGS.logdir = tempfile.mkdtemp()
    bayesian_rnn_eager_main.main([])
    self.assertNotEmpty(os.listdir(FLAGS.logdir))

    FLAGS.job = "predict_val"
    bayesian_rnn_eager_main.main([])
    self.assertNotEmpty(os.listdir(FLAGS.predict_dir))

  @flagsaver.flagsaver
  def testBayesianRNNEagerMainDeterministic(self):
    FLAGS.uncertainty_embeddings = False
    FLAGS.uncertainty_rnn = False
    FLAGS.uncertainty_hidden = False
    FLAGS.uncertainty_output = False
    FLAGS.uncertainty_biases = False

    bayesian_rnn_eager_main.main([])
    self.assertNotEmpty(os.listdir(FLAGS.logdir))
    self.assertNotEmpty(os.listdir(FLAGS.model_dir))
    self.assertEmpty(os.listdir(FLAGS.predict_dir))


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
