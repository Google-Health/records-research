# Deep State-Space Generative Model For Correlated Time-to-Event Predictions

This repository contains code implementing the model in the work of “Deep
State-Space Generative Model For Correlated Time-to-Event Predictions” (Xue, et
al. 2020). Code written using Python 3.0 and Tensorflow 1.12. NOTE: The code
provided here is not currently executable due to reliance on internal Google
utilities. In particular, the data processing pipeline is not included, but has
been described in detail in
[Rajkomar et al., 2018](https://arxiv.org/abs/1801.07860).

This is not an officially supported Google product.

## File Structure

*   [config_utils.py]. Configuration utilities.
*   [data_provider.py]. Data provider which prepares tensor inputs based on
    tensorflow slim library.
*   [datasets.py]. A slim-style dataset for clinical time series dataset.
*   [experiment.py]. Functions to build a tf.contrib.learn.Experiment.
*   [experiment_config.proto]. Configuration of the experiment and model in
    protobuf.
*   [models.py]. Main models.
*   [modules.py]. Modules used in models.
*   [multi_head_for_survival.py]. Head class for multi-task survival analysis to
    be used in tf estimator framework.
*   [sequence_heads.py]. Head class for sequence forecasting tasks.
*   [survival_heads.py]. Head class for survival analysis of a single event.
*   [survival_util.py]. Survival analysis computation utils.
*   [train.py]. Main executable for training and eval.
