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

from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from sklearn.metrics import roc_auc_score


def fit(train_sl, task, X, Y, Xv, Yv):
    best_c = None
    best_score = 0
    best_model = None

    #for K in [1*RBF(), 1*DotProduct(), 1*Matern(), 1*RationalQuadratic(), 1*WhiteKernel()]:
    for K in [1*RationalQuadratic()]:
        m = GPC(kernel=K, n_jobs=-1)
        m.fit(X, Y)
        Pv = m.predict_proba(Xv)[:,1]
        score = roc_auc_score(Yv, Pv)
        print('Fitted GPC with K:', K, 'AUC:', score)
        if score > best_score:
            best_score = score
            best_model = m
            best_c = K

    print('Best K:', best_c)
    return best_model
