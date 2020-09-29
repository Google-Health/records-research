# Pretrained hierarchical LSTMs for patient outcome prediction

This repository contains code implementing pretrained hierarchical LSTMs for
outcome prediction from electronic health records, as described in ["Improved
Patient Classification with Language Model Pretraining Over Clinical
Notes"](https://arxiv.org/abs/1909.03039) (Kemp, Rajkomar & Dai 2019). Code
written using Python 2.7 and Tensorflow 1.12.

The code sample provided here is *not executable*. We illustrate the core
implementation of our model architecture, but parts of this code reference
internal Google utilities, many of which rely on C++ implementations of custom
Tensorflow ops. For these, we describe the functionality and the expected inputs
and outputs.

Preprocessed Tensorflow SequenceExamples for MIMIC-III, used to produce the results in the above paper, are available for download by credentialed users on [Physionet](https://physionet.org/content/mimic-seqex). The format and methodology of our data pipeline, which prepares the
model inputs, have been described in detail in previous work
([“Scalable and accurate deep learning with electronic health records"](https://www.nature.com/articles/s41746-018-0029-1),
Rajkomar et al., NPJ Digital Medicine 2018).

This is not an officially supported Google product.

## Overview and Usage Guidelines

Our model requires the following inputs:

*   `sequence_features`: a dictionary mapping from a longitudinal feature name
    to a `tf.SparseTensor` containing all tokens of that feature type for a
    batch of inputs. The axes of the `SparseTensor` are of the form `[batch,
    time_step, token]`, representing which example in the batch each token
    belongs to, at what time it occurs, and where it occurs in the list of
    tokens at that time step, respectively.

*   `dense_features`: a dictionary mapping from strings `'dense_name'`,
    `'dense_value'`, `'dense_unit'` to three `tf.SparseTensors` containing the
    name, numerical value, and unit for all measurements of continuous features
    in the batch of inputs. Axes are the same as for `sequence_features`.

*   `sequence_lengths`: a `tf.Tensor` of dimension `[batch size]`, representing
    the length of each example sequence in the batch.

*   `delta_time`: a `tf.Tensor` with axes `[batch, time step]`, representing the
    time (in seconds before prediction time) associated with each discrete time
    step for each example.

*   `labels`: a `tf.Tensor` with dimensions `[batch size, number of classes]`,
    representing the true labels for each example.

We implement the following three classes to run our models:

*   `LSTM`: contains a Tensorflow implementation of an LSTM with options for
    dropout, variational dropout, and Zoneout.

*   `InputTransformer`: prepares and formats the inputs for the model, including
    embedding discrete features, normalizing continuous features, bagging, and
    running an LSTM over the notes feature for hierarchical models.

*   `EHRPredictionModel`: runs an LSTM over the transformed inputs to generate
    predictions, and computes the losses for training and pretraining.

The `EHRPredictionModel` class implements the user-facing API. To train the
model, simply call `EHRPredictionModel.Prediction` on the inputs to generate
`logits` and, if training a hierarchical model, `pretraining_inputs`.
Subsequently, call `EHRPredictionModel.Loss(logits, labels)` to generate the
loss for training, or call `EHRPredictionModel.PretrainLoss(pretraining_inputs,
sequence_features)` to generate the loss for pretraining. `loss` or
`pretrain_loss` can then be minimized using standard Tensorflow methods, e.g.
calling `optimizer.minimize(loss)` with a `tf.train.Optimizer` of choice.

## Description of Internal Utilities

In this code, we reference internal utilities under the module name `g_ops`.
Although we do not provide implementation details for these utilities, we
describe their function here.

```
bagging(seq_feature, bagging_ids, seq_lengths):

    Groups the values given in the SparseTensor seq_features into bags according
    to the corresponding bagging_ids, for embedding and aggregation.

    Args:
      seq_feature: a SparseTensor of feature IDs with shape [batch_size,
        max_time, max_num_tokens].
      bagging_ids: a Tensor of shape [batch_size, max_time] with the bag IDs at
        each timestep.
      seq_lengths: a Tensor of shape [batch_size] with the lengths of each
        sequence in the batch along the time axis.

    Returns: a SparseTensor of shape [batch_size, max_num_bags, max_time *
      max_num_tokens]. Embeddings will then be aggregated within bags by
      tf.safe_embedding_lookup_sparse.

bagging_feature_to_ids(bagging_feature, seq_lengths):

    Converts the bagging_feature (generally time or similar) into IDs to be used
    in g_ops.bagging.

    Args:
      bagging_feature: a Tensor of shape [batch_size, max_time] with the time
        values to use for bagging.
      seq_lengths: a Tensor of shape [batch_size] with the lengths of each
        sequence in the batch along the time axis, before bagging.

    Returns:
      bagging_ids: a Tensor of shape [batch_size, max_time] with the IDs derived
        from bagging_feature.
      unique_time: a Tensor of shape [batch_size, max_time] with the time value
        chosen to represent each bag.
      bagging_seq_lengths: a Tensor of shape [batch_size] with the lengths of
        each sequence in the batch along the time axis, after bagging.

batch_by_index_pair(data, indices):

    Batches elements of 'data' according to the leading pair of 'indices'. We
    use this function to reshape a matrix of word embeddings into a batch of
    embedded sequences, where each sequence corresponds to one note.

    Args:
      data: a tensor with shape [num_tokens, embed_dim].
      indices: a tensor with shape [num_tokens, 3] containing (batch, time,
        token) indices for each token.

    Returns:
      batched_data: a tensor of shape [batch_size, seq_len, embed_dim], where:
        - batch_size is the number of unique (indices[0], indices[1]) pairs.
        - seq_len is the maximum batched sequence length, i.e. the maximum
          number of unique indices[2] for any (indices[0], indices[1]) pair.
        - Sequences shorter than length seq_len are zero-padded.
      batched_seq_lengths: a Tensor of shape [batch_size] with the sequence
        lengths after batching.
      orig_indices: a Tensor of shape [batch_size, 2] with the original
        (indices[0], indices[1]) pairs for each batch entry.

load_embedding_config_proto(embed_config_path):

    Loads a protocol buffer, a Google format for serializing data, for storing
    vocab size and embedding dimension by feature name. This is mostly a
    convenience - the code also shows how to specify these manually.

    Args:
      embed_config_path: a filepath to the config.

    Returns: the config object, which behaves like a dict by feature name.

normalize_dense(dense_features, delta_time, config_path):

    Standardizes dense values to Z-scores according to stats provided in
    config_path.  Outliers are capped to a Z-score of +/- 10. These values are
    then bagged and reshaped for concatenation with the other sequence features
    used for prediction.

    Args:
      dense_features: three SparseTensors containing names, values, and units
        for continuous features, as described above.
      delta_time: a Tensor of shape [batch_size, max_time].
      config_path: filepath to the config containing summary stats for each
        feature.

    Returns: a Tensor of shape [batch_size, max_time, num_dense_features] with
      the standardized values.

sequence_softmax(seq_inputs, seq_lengths):

    Computes the softmax along the time axis for a zero-padded sequence tensor,
    over valid (non-padding) entries only.

    Args:
      seq_inputs: a Tensor of shape [batch_size, max_time, ...].
      seq_lengths: a Tensor of shape [batch_size] with the length of each
        sequence.

    Returns: a Tensor of the same shape as seq_inputs, where each value is
      replaced by its softmax score along the time axis (padding entries remain
      zero).

sparse_nd_select_last_k(sparse_features, max_length):

    Retains the last max_length nonzero elements in each slice of
    sparse_features along the first axis (based on row-major ordering). In our
    models, this corresponds to keeping the last max_length words from all notes
    in each patient record.

    Args:
      sparse_features: a SparseTensor of IDs with shape [batch_size, max_time,
        max_num_tokens]. sparse_feature.indices should be in row-major order.
      max_length: the number of IDs to retain per slice.

    Returns: a SparseTensor containing the last max_length IDs per value of
      sparse_features.indices[:, 0].
```
