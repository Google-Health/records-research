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

from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import roc_auc_score

def fit(train_sl, task, X, Y, Xv, Yv):
    best_c = None
    best_score = 0
    best_model = None
    for c in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        m = MLP(hidden_layer_sizes=(256, 256, 256, 256), alpha=c,
                early_stopping=True,
                learning_rate='adaptive',
                batch_size=128)
        m.fit(X, Y)
        Pv = m.predict_proba(Xv)[:,1]
        score = roc_auc_score(Yv, Pv)
        print('Fitted model with C:', c, 'AUC:', score)
        if score > best_score:
            best_score = score
            best_model = m
            best_c = c

    print('Best C:', best_c, 'AUC:', best_score)
    return best_model
