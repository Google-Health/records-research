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

"""Tests for the BayesianRNN model."""
import edward2 as ed
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

import bayesian_rnn_model


# TODO(dusenberrymw): Add open-source tests for the full models.
class BayesianRNNTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testClipGradient(self):
    shape = (2, 3)
    clip_norm = 1e-3
    with tf.GradientTape(persistent=True) as tape:
      x = tf.zeros(shape)
      tape.watch(x)
      y1 = tf.identity(x)
      y2 = bayesian_rnn_model._clip_gradient(x, clip_norm)
    self.assertAllEqual(self.evaluate(x), self.evaluate(y2))

    dy1dx = tape.gradient(y1, x)
    dy2dx = tape.gradient(y2, x)
    dx_unclipped = tf.constant(1.0, shape=shape)
    dx_clipped = (dx_unclipped * clip_norm /
                  tf.maximum(tf.linalg.global_norm([dx_unclipped]), clip_norm))
    self.assertEqual(dy2dx.shape, list(shape))
    self.assertAllEqual(self.evaluate(dy1dx), self.evaluate(dx_unclipped))
    self.assertAllClose(self.evaluate(dy2dx), dx_clipped)

    grad_ys = tf.constant(1e-4, shape=shape)
    dy2dx = tape.gradient(y2, x, output_gradients=grad_ys)
    # The l2 norm is sqrt(6 * (1e-4)^2) = 2.44e-4, which is less than 1e-3, so
    # no clipping should occur.
    self.assertAllClose(self.evaluate(dy2dx), self.evaluate(grad_ys))

  @test_util.run_in_graph_and_eager_modes
  def testLSTMCellReparameterizationGradClipped(self):
    clip_norm = 1e-3
    shape = (2, 4)
    with tf.GradientTape(persistent=True) as tape:
      x = tf.zeros(shape)
      tape.watch(x)
      cell1 = ed.layers.LSTMCellReparameterization(10)
      cell2 = bayesian_rnn_model.LSTMCellReparameterizationGradClipped(
          clip_norm, 10)
      h0, c0 = cell1.get_initial_state(x)
      tape.watch(h0)
      tape.watch(c0)
      state = (h0, c0)
      y1, _ = cell1(x, state)
      y2, _ = cell2(x, state)

    self.evaluate(tf1.global_variables_initializer())

    grad_ys = tf.constant(1e6, shape=y1.shape)
    dy1dx = tape.gradient(y1, x, output_gradients=grad_ys)
    dy2dx = tape.gradient(y2, x, output_gradients=grad_ys)
    res1 = self.evaluate(tf.linalg.norm(dy1dx))
    res2 = self.evaluate(tf.linalg.norm(dy2dx))
    self.assertNotAllClose(res1, res2)
    self.assertGreater(res1, clip_norm)
    self.assertAllClose(res2, clip_norm)

    dy1dh0 = tape.gradient(y1, h0, output_gradients=grad_ys)
    dy2dh0 = tape.gradient(y2, h0, output_gradients=grad_ys)
    res1 = self.evaluate(tf.linalg.norm(dy1dh0))
    res2 = self.evaluate(tf.linalg.norm(dy2dh0))
    self.assertNotAllClose(res1, res2)
    self.assertGreater(res1, clip_norm)
    self.assertAllClose(res2, clip_norm)

    dy1dc0 = tape.gradient(y1, c0, output_gradients=grad_ys)
    dy2dc0 = tape.gradient(y2, c0, output_gradients=grad_ys)
    res1 = self.evaluate(tf.linalg.norm(dy1dc0))
    res2 = self.evaluate(tf.linalg.norm(dy2dc0))
    self.assertNotAllClose(res1, res2)
    self.assertGreater(res1, clip_norm)
    self.assertAllClose(res2, clip_norm)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
