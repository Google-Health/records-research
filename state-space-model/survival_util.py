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

"""Utility functions for Survival Analysis Models."""
import abc
import modules

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

INITIAL_LN_RATE = -13
UNITS_IN_HR = 60 * 60
DTYPE = tf.float32
MAX_SLOT = 20*24


class LeakyReLU(tfb.Bijector):
  """Multiplying by alpha causes a contraction in volume."""

  def __init__(self, alpha=0.5, validate_args=False, name='leaky_relu'):
    super(LeakyReLU, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name,
        is_constant_jacobian=True)
    self.alpha = alpha

  def _forward(self, x):
    return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

  def _inverse(self, y):
    return tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)

  def _inverse_log_det_jacobian(self, y):
    idt = tf.ones_like(y)
    jacobian_inv = tf.where(tf.greater_equal(y, 0), idt, 1.0 / self.alpha * idt)
    log_abs_det_jacobian_inv = tf.log(tf.abs(jacobian_inv))
    return log_abs_det_jacobian_inv


class SurvivalModel(object):
  """Base class for survival models."""

  @abc.abstractproperty
  def params(self):
    """Returns the parameters of this model."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def log_pdf(self, t):
    """Returns the log of probablity density function."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def log_survival_func(self, t):
    """Returns the log of survival function."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def cdf(self, t):
    """Cumulative incidence/density functions."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def hazard_rate(self, t):
    """Instantaneous rate of event occurrence in 1/sec ."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def probability(self):
    """Expected incidence probability."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def predicted_time(self):
    """Predicted Event Time in sec."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def probability_within_window(self, window_start_t, window_end_t):
    """Predicted Event Probability within window."""
    raise NotImplementedError('Calling an abstract method.')


class ParametricExponentialSurvivalModel(SurvivalModel):
  """Parametric survival model based on exponential distribution."""

  def __init__(self, params, labels, event_index, model_hparams=None):
    del model_hparams
    del labels
    del event_index
    params_shape = array_ops.shape(params)
    assert_rank = check_ops.assert_rank_at_least(
        params,
        2,
        data=[params_shape],
        message='Exponential model params shape must be [batch_size, 1]')

    with ops.control_dependencies([assert_rank]):
      # TODO(yuanxue,gafm): Experiment with tf.softplus.
      self._rate_param = tf.exp(params + INITIAL_LN_RATE)
      self._distribution = tfp.distributions.Exponential(rate=self._rate_param)

  @property
  def params(self):
    return self._rate_param

  def log_pdf(self, t):
    """Log of PDF of exponential distribution.

       PDF(t, lambda) = lambda * exp(-lambda * t)
       log(PDF(t, lambda)) = log(lambda) - lambda * t
    Args:
      t: time instance where the function is evaluated: scalar or tensor of
        [batch_size, 1]

    Returns:
      Value of log of PDF of exponential distribution: tensor of [batch_size, 1]
    """
    return self._distribution.log_prob(t)

  def log_survival_func(self, t):
    """Log of survival function of exponential distribution.

       S(t, lambda) = 1- CDF(t, lambda) = exp(-lambda * t)
       log(S(t, lambda)) = - lambda*t
    Args:
      t: time instance where the function is evaluated.
    Returns:
      Value of log of survival function of exponential distribution.
    """
    # Note that in the tfp implementation of Exponential distribution,
    # log_prob is more stable than log_survival_function.
    return self._distribution.log_prob(t) - tf.log(self._rate_param)

  def cdf(self, t):
    """Cumulative incidence functions: probablity of event happening before t.

       F(t, lambda) = CDF(t, lambda)  = 1- exp(-lambda * t)
    Args:
      t: time instance where the function is evaluated.

    Returns:
      Value of log of survival function of exponential distribution.
    """
    return 1 - tf.exp(-self._rate_param * t)

  def hazard_rate(self, t):
    """Instantaneous rate of event occurrence."""
    return self._rate_param

  def probability(self):
    """Expected incidence probability."""
    return self._rate_param

  def predicted_time(self):
    return tf.div(tf.reciprocal(self._rate_param), UNITS_IN_HR)

  def probability_within_window(self, window_start_t, window_end_t):
    return tf.exp(-self._rate_param * window_start_t) - tf.exp(
        -self._rate_param * window_end_t)


class CoxSurvivalModel(SurvivalModel):
  """Cox proportional harzard survival model."""

  def __init__(self, params, labels, event_index, model_hparams=None):
    del model_hparams
    del labels
    del event_index
    params_shape = array_ops.shape(params)
    assert_rank = check_ops.assert_rank_at_least(
        params,
        2,
        data=[params_shape],
        message='Cox model params shape must be [batch_size, num_feature]')

    with ops.control_dependencies([assert_rank]):
      logits_shape = params.get_shape()[1]
      # We assume the base rate is constant in this implementation:
      # lambda = lambda_0 * exp(X * weights) = exp (bias + logits * weights)
      with tf.variable_scope('logit_to_parameter', reuse=tf.AUTO_REUSE):
        self._weights = tf.get_variable(
            'weights', [logits_shape, 1],
            initializer=tf.initializers.truncated_normal(0, 0.01))
        self._bias = tf.get_variable(
            'bias', [1],
            initializer=tf.initializers.truncated_normal(0.01, 0.01))
        weighted_logits = tf.matmul(params, self._weights) + self._bias
        self._rate_param = tf.exp(weighted_logits)
        self._distribution = tfp.distributions.Exponential(
            rate=self._rate_param)

  @property
  def params(self):
    return self._rate_param

  def log_pdf(self, t):
    """Log of PDF of exponential distribution.

       PDF(t, lambda) = lambda * exp(-lambda * t)
       log(PDF(t, lambda)) = log(lambda) - lambda * t
    Args:
      t: time instance where the function is evaluated.

    Returns:
      Value of log of PDF of exponential distribution.
    """
    return self._distribution.log_prob(t)

  def log_survival_func(self, t):
    """Log of survival function of exponential distribution.

       S(t, lambda) = 1- CDF(t, lambda) = exp(-lambda * t)
       log(S(t, lambda)) = - lambda*t
    Args:
      t: time instance where the function is evaluated.

    Returns:
      Value of log of survival function of exponential distribution.
    """
    # Note that in the tfp implementation of Exponential distribution,
    # log_prob is more stable than log_survival_function.
    return self._distribution.log_prob(t) - tf.log(self._rate_param)

  def cdf(self, t):
    """Cumulative incidence functions: probablity of event happening before t.

       F(t, lambda) = CDF(t, lambda)  = 1- exp(-lambda * t)
    Args:
      t: time instance where the function is evaluated.

    Returns:
      Value of log of survival function of exponential distribution.
    """
    return 1 - tf.exp(-self._rate_param * t)

  def hazard_rate(self, t):
    """Instantaneous rate of event occurrence."""
    return self._rate_param

  def probability(self):
    """Expected incidence probability."""
    return self._rate_param

  def predicted_time(self):
    return tf.reciprocal(self._rate_param)

  def probability_within_window(self, window_start_t, window_end_t):
    return tf.exp(-self._rate_param * window_start_t) - tf.exp(
        -self._rate_param * window_end_t)


class StateSpaceSurvivalModel(SurvivalModel):
  """Discrete time survival model generated from state space."""

  def __init__(self, params, labels, event_index, model_hparams=None):
    del labels
    del event_index
    self._model_hparams = model_hparams
    # init_state is encoded state at trigger time passed from params(logits).
    init_state = params
    self._slot_size_hr = model_hparams.da_sslot
    self._time_len = model_hparams.da_tlen
    batch_size = init_state.get_shape().as_list()[0]
    # params is the input x with shape [batch_size, num_features]
    if model_hparams.reuse_encoding:
      self._tag = ''
    else:
      self._tag = 'encode'
    hazard_at_trigger = tf.zeros([batch_size, 1])
    # forecast_hazard shape [da_tlen, batch_size, 1]
    forecast_hazard = self._forecast_hazard(init_state, hazard_at_trigger)
    # self._hazard_tensor shape shape [TIME_LEN, batch_size]
    self._hazard_tensor = tf.squeeze(forecast_hazard, axis=-1)
    self._hazard_tensor = tf.clip_by_value(
        self._hazard_tensor, 1e-20, 0.99999, name=None)

    #   lmbda = tf.clip_by_value(lmbda, 1e-20, 0.99999, name=None)

  def _forecast_hazard(self, init_state, hazard_at_trigger):
    """Forecast hazard."""
    tf.logging.info(init_state)
    dummy_input = tf.zeros([self._model_hparams.da_tlen])
    tf.logging.info(dummy_input)
    tf.logging.info(hazard_at_trigger)

    # Get obs values at trigger time as base for delta prediction.
    forecast_hazard, _ = tf.scan(
        self._state_tran_and_hazard_emission_step_fn,
        dummy_input,  # not used just control the steps.
        initializer=(hazard_at_trigger, init_state),  # state at trigger time
        parallel_iterations=10,
        name='hazard_forecast_scan')
    # forecast_hazard shape [da_tlen, batch_size, 1]
    return forecast_hazard

  def _state_tran_and_hazard_emission_step_fn(
      self, previous_output, current_input):
    """State transition and hazard rate emission in a single step."""
    del current_input
    previous_hazard, previous_state = previous_output
    current_state = modules._state_tran_module(  # pylint: disable=protected-access
        previous_state, self._model_hparams.ds_state,
        self._model_hparams.stran_nmlp, self._model_hparams.stran_smlp,
        self._tag)

    if self._model_hparams.forecast_interv:
      # interv forecast is performed and needs to be incorporated into state
      # transiention.
      current_interv = modules._interv_forecast_module(  # pylint: disable=protected-access
          previous_state, len(self._model_hparams.intervention_codes),
          self._model_hparams.interv_nmlp, self._model_hparams.interv_smlp,
          self._tag)
      # next_interv is applied to next_state.
      current_state = current_state + modules._control_tran_module(  # pylint: disable=protected-access
          current_interv, self._model_hparams.ds_state,
          self._model_hparams.ctran_nmlp, self._model_hparams.ctran_smlp,
          self._tag)

    current_emission = modules._hazard_emission_module(  # pylint: disable=protected-access
        current_state, self._model_hparams.hazard_nmlp,
        self._model_hparams.hazard_smlp, self._tag)
    if self._model_hparams.forecast_delta:
      current_hazard = current_emission + previous_hazard
    else:
      current_hazard = current_emission
    tf.logging.info(current_hazard)
    tf.logging.info(current_state)

    return (current_hazard, current_state)

  def _bucketize_t(self, t):
    """Turn time instance tensor t in sec to a time slot."""
    # t shape [batch_size]
    time_slot = tf.cast(tf.div(t, self._slot_size_hr * UNITS_IN_HR), tf.int32)

    capped_time_slot = tf.where(
        tf.greater_equal(time_slot, self._time_len),
        tf.fill(tf.shape(time_slot), self._time_len),
        tf.add(time_slot, tf.ones_like(time_slot)))
    # position in #hrs in the last time slot.
    last_slot_hr = tf.where(
        tf.greater_equal(time_slot, self._time_len),
        time_slot - self._time_len + 1,
        tf.zeros_like((time_slot)))
    # capped_time_slot shape [batch_size], each value range [1, self._time_len]
    return capped_time_slot, last_slot_hr

  def _from_slot_to_time_range(self, slot):
    """Turn time slot slot (scalar) to a time range [t_start, t_end)."""
    t_start = slot * self._slot_size_hr * UNITS_IN_HR
    t_end = (slot + 1) * self._slot_size_hr * UNITS_IN_HR
    return t_start, t_end

  def params(self):
    return tf.transpose(self._hazard_tensor)

  def log_pdf(self, t):
    """Log of PDF of the distribution.

    Args:
      t: time instance. Tensor of shape [batch_size]. TODO--> [batch_size, 1]

    Returns:
      Value of log of PDF. Tensor of shape [batch_size, 1].
    """
    t = tf.squeeze(t)
    t, last_slot_hr = self._bucketize_t(t)

    ones = tf.fill(tf.shape(t), 1)
    # shape [batch_size, TIME_LEN]
    seq_mask_t_1 = tf.sequence_mask(
        tf.cast(t - ones, tf.int32), maxlen=self._time_len)
    # shape [TIME_LEN, batch_size]
    lambda_tensor = self._hazard_tensor

    # shape [batch_size, TIME_LEN], multiply supports broadcast.
    lambda_tensor_t_1 = tf.multiply(
        tf.transpose(lambda_tensor), tf.cast(seq_mask_t_1, tf.float32))

    # shape [batch_size, TIME_LEN]
    seq_mask_t = tf.sequence_mask(tf.cast(t, tf.int32), maxlen=self._time_len)
    # shape [batch_size, TIME_LEN]
    mask_at_t = tf.logical_xor(seq_mask_t, seq_mask_t_1)

    # shape [batch_size]
    selected_lambda_tensor_at_t = tf.boolean_mask(
        tf.transpose(lambda_tensor), mask_at_t)
    # selected_lambda_tensor_at_t = tf.Print(
    #    selected_lambda_tensor_at_t, [selected_lambda_tensor_at_t],
    #    'selected_lambda_tensor_at_t',
    #    summarize=self._time_len)

    # shape [batch_size, 1]
    result = tf.reduce_sum(
        tf.log(1 - lambda_tensor_t_1), axis=-1, keepdims=True) + tf.log(
            tf.reshape(selected_lambda_tensor_at_t, [-1, 1]))

    if self._model_hparams.last_slot_loss:
      # last_slot_hr is the position of t in terms of #hrs in the last slot.
      # tf.multiply performs element-wise multiplication along batch dimension.
      result = result + tf.reduce_sum(
          tf.multiply(tf.log(1-lambda_tensor[self._time_len-1]),
                      tf.cast(last_slot_hr, tf.float32)),
          axis=-1, keepdims=True)

    return result

  def log_survival_func(self, t):
    """Log of survival function.

       log S(t, lambda(k)) = sum_{k=1}^{t} log(1 -lambda(k))
    Args:
      t: time instance in sec. scalar or Tensor of shape [batch_size].

    Returns:
      Value of log of survival function. Tensor of shape [batch_size, 1].
    """
    # shape [batch_size, TIME_LEN]
    t = tf.squeeze(t)
    t, last_slot_hr = self._bucketize_t(t)

    seq_mask = tf.sequence_mask(tf.cast(t, tf.int32), maxlen=self._time_len)
    # tf.logging.info(seq_mask)
    # shape [TIME_LEN, batch_size]
    lambda_tensor = self._hazard_tensor
    # tf.logging.info(lambda_tensor)
    # shape [batch_size, TIME_LEN], multiply supports broadcast.
    active_lambda_tensor = tf.multiply(
        tf.transpose(lambda_tensor), tf.cast(seq_mask, tf.float32))
    # tf.logging.info(active_lambda_tensor)
    result = tf.reduce_sum(
        tf.log(1 - active_lambda_tensor), axis=-1, keepdims=True)

    if self._model_hparams.last_slot_loss:
      # last_slot_hr is the position of t in terms of #hrs in the last slot.
      # tf.multiply performs element-wise multiplication along batch dimension.
      result = result + tf.reduce_sum(
          tf.multiply(tf.log(1-lambda_tensor[self._time_len-1]),
                      tf.cast(last_slot_hr, tf.float32)),
          axis=-1, keepdims=True)

    return result

  def cdf(self, t):
    """Cumulative incidence functions: probablity of event happening before t.

       F(t, lambda) = CDF(t, lambda)  = 1- S(t, lambda)
    Args:
      t: time instance. Tensor of shape [batch_size.

    Returns:
      Value of CDF.
    """
    return 1 - tf.exp(self.log_survival_func(t))

  def hazard_rate(self, t):
    """Hazard rate at time t."""
    # t is a scalar.
    t = self._bucketize_t(t)
    seq_mask_t = tf.sequence_mask(tf.cast(t, tf.int32), maxlen=self._time_len)
    ones = tf.fill(tf.shape(t), 1)
    seq_mask_t_1 = tf.sequence_mask(
        tf.cast(t - ones, tf.int32), maxlen=self._time_len)
    mask_at_t = tf.logical_xor(seq_mask_t, seq_mask_t_1)

    # shape [TIME_LEN, batch_size]
    lambda_tensor = self._hazard_tensor
    # tf.logging.info(lambda_tensor)

    # shape [batch_size]
    selected_lambda_tensor_at_t = tf.boolean_mask(
        tf.transpose(lambda_tensor), mask_at_t)
    # tf.logging.info(selected_lambda_tensor_at_t)

    # shape [batch_size, 1]
    return tf.reshape(selected_lambda_tensor_at_t, [-1, 1])

  def probability_within_window(self, window_start_t, window_end_t):
    """Predicted event probablity within the time window.

    Args:
      window_start_t: time instance of time window.
      window_end_t: time instance of time window.

    Returns:
      Value of Predicted event probablity within window.
    """
    shape_tensor = self.log_survival_func(window_end_t)
    if window_start_t == 0:
      window_survival_start = tf.fill(tf.shape(shape_tensor), 1.0)
    else:
      window_survival_start = tf.exp(self.log_survival_func(window_start_t))

    if window_end_t > self._time_len * self._slot_size_hr * UNITS_IN_HR:
      window_survival_end = tf.fill(tf.shape(shape_tensor), 0.0)
    else:
      window_survival_end = tf.exp(self.log_survival_func(window_end_t))

    window_p = window_survival_start - window_survival_end
    return window_p

  def probability(self):
    """Predicted event probablity, will be used for C-Index computation.

    Returns:
      Value of inverse of expected event time.
    """
    return tf.reciprocal(self.predicted_time())

  def predicted_time(self):
    """Predicted event time in hr."""
    # TODO(yuanxue): need a close form for MAX_SLOT-> infinity.
    survival_list = [
        tf.exp(
            self.log_survival_func(self._from_slot_to_time_range(time_slot)[1]))
        * self._slot_size_hr for time_slot in range(MAX_SLOT)
    ]
    survival_time_hr = tf.reduce_sum(
        tf.concat(survival_list, axis=1), axis=-1, keep_dims=True)
    #    survival_time_hr = tf.Print(
    #        survival_time_hr, [survival_time_hr],
    #        'survival_time_hr',
    #        summarize=32)
    return survival_time_hr


def negative_log_likelihood_loss(censored, log_pdf_value, log_survival_value):
  """Compute Negative log likelihood, which can be used for training loss.

  Args:
    censored: True, if the event is censored (i.e., not observed).
    log_pdf_value: Log of pdf, representing the likelihood of event observed.
    log_survival_value: Log of survival function, representing the likelihood of
      event censored.

  Returns:
    Value of Negative log likelihood.
  """
  return -tf.where(censored, log_survival_value, log_pdf_value)


REGISTERED_SURVIVAL_MODEL = {
    'exponential': ParametricExponentialSurvivalModel,
    'cox': CoxSurvivalModel,
    'state': StateSpaceSurvivalModel,
}
