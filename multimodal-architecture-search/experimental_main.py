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

"""Main executable."""

import experiment
from multimodal_transformer_model import MultimodalTransformerModel
import tensorflow.compat.v1 as tf


def main(unused_argv):
  model_instance = MultimodalTransformerModel()
  experiment.run(
      model_instance=model_instance,
      num_train_steps=100,
      num_eval_steps=10,
      save_checkpoints_secs=100)


if __name__ == '__main__':
  tf.app.run()
