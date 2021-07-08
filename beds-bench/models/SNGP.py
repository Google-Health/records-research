# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import edward2 as ed
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import roc_auc_score

tf.random.set_seed(2020)

BATCH_SIZE = 128

class GetBest(Callback):
    """
    From https://github.com/keras-team/keras/issues/2768#issuecomment-361070688
    """

    """Get the best model at the end of training.
        # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
        # Example
                callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
                mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            print('GetBest mode %s is unknown, '
                  'fallback to auto mode.' % (mode),
                  RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs=None):
        #self.best_weights = self.model.get_weights()
        return

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            for key, val in logs.items():
                if key.startswith(self.monitor):
                    current = val
                    break
            if current is None:
                print('Can pick best model only with %s available, '
                      'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 1:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 1:
                        print('Epoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)

class SNGPDNN(tf.keras.Model):
    def __init__(self,
                 depth=8,
                 width=256,
                 activation='relu',
                 l2=5e-2,
                 gp_input_dim=128,
                 gp_hidden_dim=1024,
                 gp_scale=2,
                 gp_bias=0,
                 gp_input_normalization=True,
                 gp_cov_discount_factor=0.999,
                 gp_cov_ridge_penalty=1e-3,
                 use_spec_norm=True,
                 spec_norm_iteration=5,
                 spec_norm_bound=0.95,
                 use_gp_layer=True):
        super(SNGPDNN, self).__init__()

        D = lambda: Dense(width,
                          activation=Activation(activation),
                          kernel_regularizer=tf.keras.regularizers.l2(l2))
        GP = lambda: ed.layers.RandomFeatureGaussianProcess(1, num_inducing=gp_hidden_dim,
                                                            gp_kernel_scale=gp_scale,
                                                            gp_output_bias=gp_bias,
                                                            normalize_input=gp_input_normalization,
                                                            gp_cov_momentum=gp_cov_discount_factor,
                                                            gp_cov_ridge_penalty=gp_cov_ridge_penalty)

        block = lambda: ed.layers.SpectralNormalization(D(),
                                                        iteration=spec_norm_iteration,
                                                        norm_multiplier=spec_norm_bound) if use_spec_norm else D()
        self.dnn = Sequential([block() for _ in range(depth)])
        self.use_gp_layer = use_gp_layer
        if use_gp_layer:
            self.proj_layer = Dense(gp_input_dim,
                                    kernel_initializer='random_normal',
                                    use_bias=False,
                                    trainable=False)
            self.gp_layer = GP()
        else:
            self.last_layer = Dense(1)
        pass

    def call(self, inputs):
        x = self.dnn(inputs)
        if self.use_gp_layer:
            logits, _ = self.gp_layer(self.proj_layer(x))
        else:
            logits = self.last_layer(x)
        return tf.keras.activations.sigmoid(logits)

    def predict_proba(self, inputs, **kwargs):
        probs = self.call(inputs).numpy()
        return np.append(1-probs, probs, axis=1)

    def fit(self, *args, **kwargs):
        callbacks = [GetBest(monitor='val_auc', verbose=0, mode='max'),
                     EarlyStopping(monitor="val_loss", min_delta=1e-3,
                                   patience=5, verbose=0)]
        return super().fit(*args, callbacks=callbacks, **kwargs)

class KerasClassifierLOSS(KerasClassifier):
    def nll(self, X, Y):
        probs = self.model(X).numpy().flatten()
        loss = Y * np.log(probs) + (1-Y) * np.log(1-probs)
        return -np.mean(loss)


    def score(self, X, Y):
        probs = self.model(X).numpy().flatten()
        return roc_auc_score(Y, probs)

    def fit(self, *args, **kwargs):
        super(KerasClassifierLOSS, self).fit(*args, **kwargs)

def create_model(depth, l2, width, activation):
    model = SNGPDNN(depth=depth, width=width, l2=l2, activation=activation)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])
    return model

def fit(train_sl, task, X, Y, Xv, Yv):
    param_grid = dict(l2=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3],
                      depth=[2, 3, 4, 5],
                      width=[64, 128, 256, 512],
                      batch_size=[32, 64, 128, 256],
                      activation=['tanh', 'relu', 'sigmoid'],
                      epochs=[40])
    builder = KerasClassifierLOSS(create_model, verbose=0)
    best_model = RandomizedSearchCV(builder, param_grid, cv=2, n_jobs=10, n_iter=128, random_state=2020)
    config = best_model.fit(X, Y, validation_data=(Xv, Yv))

    print('Best AUC:', roc_auc_score(Yv, best_model.predict_proba(Xv)[:,1]))
    print('Best config:', best_model.best_params_)
    return best_model

if __name__ == '__main__':
    M = 1024
    N = 128
    X = np.random.random((M, N))
    Y = np.random.binomial(1, 0.1, size=M)
    Xv = np.random.random((M, N))
    Yv = np.random.binomial(1, 0.1, size=M)

    best_model = fit_model(X, Y, Xv, Yv)
