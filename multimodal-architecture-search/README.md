# MUFASA: Multimodal Fusion Architecture Search for Electronic Health Records

This repository contains code implementing multimodal fusion architecture for
prediction tasks on electronic health records, as described in
["MUFASA: Multimodal Fusion Architecture Search for Electronic Health Records"](https://arxiv.org/abs/2102.02340)
(Xu, So & Dai 2021). Codes are written using Python 3.6. Tensorflow 2.4.1, and
[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) library.

In order to make the code executable, we use the simulated data because we can
not directly share the EHR data and all relevant preprocessing modules. The
focus of this code repository is to show the multimodal fusion architecture for
EHR that proesses the categorical features, continuous features and clinical
notes differently.

For people are interested in EHR preprocessing, we illustrate the core
implementation in
[this repo](https://github.com/Google-Health/records-research/tree/master/clinical-notes-prediction).
In addition, the preprocessed Tensorflow SequenceExamples for MIMIC-III, used to
produce the results in the above paper, are available for download by
credentialed users on [Physionet](https://physionet.org/content/mimic-seqex).

This is not an officially supported Google product.

## File Structure and Overview

Our repo contains the following files:

*   `experimental_main.py`: This the main executable file.

*   `experiment.py`: This files contains the basic code for training and
    evaluation. make_training_spec_fn() creates a DistributedTrainingSpec for
    training. In input_fn(), we generated the simulated training data. The input
    data in the shape of \[batch_size, seq_len, hidden_dim\]. It assumes all
    sequences has length 10, for the hidden dimension, dimensions \[0:8\] are
    for categorical features; Dimensions \[8:24\] are for continuous feature;
    Dimensions \[24:56\] are for the clinical notes. They are for the
    illustative purpose.

*   `multimodal_transformer_model.py`: This file contains the main model
    architecture. Specifically, mufasa_model() contains the MUSAFA model
    architecture that processes a list of input tensors in the order of
    \[categorical features, continuous features, clinical notes\]. Each tensor
    is in the shape of \[batch_size, seq_len, feature_hidden_dim\]. On the high
    level, we input three sets of sequence features into the mufasa model and
    output a single sequence output. The model behaves like an encoder and we
    use the the last state of the output as the final representation of sequence
    features. Then we concatenate the sequence feature representation with the
    context feature representation, which in shape of \[batch_size,
    hidden_size\] as our final representation for the patient. Then it goes
    through a dense layer to generate the final logits.
