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

"""Utility functions for reading and constructing experiment configuration."""
import copy
import re

import experiment_config_pb2
import tensorflow as tf

from google.protobuf import descriptor
from google.protobuf import message as proto2
from google.protobuf import text_format


def load_experiment_config(
    filename: bytes) -> experiment_config_pb2.ExperimentConfig:
  """Loads ExperimentConfig from a CNS pbtxt file.

  Args:
    filename: Pathname to config file.

  Returns:
    An ExperimentConfig proto.
  """
  f = open(filename)
  return text_format.Parse(f.read(), experiment_config_pb2.ExperimentConfig())


def merge_from_hparams(original_message: proto2.Message,
                       hparams: tf.HParams,
                       delimiter='.'):
  """Merge hparams with the original_message and return the new proto.

  Args:
    original_message: Original proto message.
    hparams: Fields to be updated. The attribute names can point to
      nested messages using a delimiter. For example,
      if the delimiter is '.', then
        'tf.HParams(**{'model.batch_size': 12})' points to the 'batch_size'
        attribute of the field 'model' in the 'original_message'.
    delimiter: The separator used to delimit nested fields. The default is a dot
      ('.') meaning that 'foo.bar' will point to `foo.bar`.

  Returns:
    Merged proto with the fields present in the hparams object overridden.

  Raises:
    AttributeError: If 'hparams' points to a nonexistent field in
      'original_message'.
  """

  message = copy.deepcopy(original_message)

  for name, value in hparams.values().items():
    sub_fields = str(name).split(delimiter)
    # Descend to the leaf node.
    sub_message = message
    for sub_field in sub_fields[:-1]:
      sub_message = getattr(sub_message, sub_field)

    leaf_field = sub_fields[-1]
    if leaf_field in sub_message.DESCRIPTOR.fields_by_name:
      field_descriptor = sub_message.DESCRIPTOR.fields_by_name[leaf_field]
      is_repeated = (
          field_descriptor.label == descriptor.FieldDescriptor.LABEL_REPEATED)
    else:
      raise AttributeError('Message %s has no field %r' % (type(message), name))

    sub_message.ClearField(leaf_field)
    if is_repeated:
      getattr(sub_message, leaf_field).extend(value)
    else:
      setattr(sub_message, leaf_field, value)

  return message


def sanitize_parameter_names(study_config):
  """Replaces dots with double underscores in `study_config`'s parameter names.

  If dots ('.') are used as field delimiters in the parameter names, we replace
  them with double underscores ('__') to accommodate the regex used by the
  tf.StudyConfiguration to parse multidimensional hyperparameter names.

  Args:
    study_config: (vizier.StudyConfiguration) A study configuration with
      a repeated field `parameter_configs` containing vizier.ParameterConfig
      proto messages.

  Returns:
    A modified study configuration, with the parameter name scrubbed of dots
    and replaced with double underscores.
  """
  for pconfig in study_config.parameter_configs:
    pconfig.name = pconfig.name.replace('.', '__')
  return study_config


def sanitize(text):
  """Makes a string suitable for a directory name.

  Args:
    text: (string) The input string to be cleaned up.

  Returns:
    (string) A sanitized version, possibly empty.
  """
  word_regex = re.compile(r'[a-zA-Z0-9]+')

  return '_'.join(word_regex.findall(text.lower()))
