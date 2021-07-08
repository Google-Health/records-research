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

from skgarden import MondrianForestClassifier as MF
from sklearn.metrics import roc_auc_score

def fit(train_sl, task, X, Y, Xv, Yv):
    best_n = None
    best_score = 0
    best_model = None
    for n in [10]: #, 50, 100]:
        m = MF(n_estimators=n, n_jobs=-1)
        print('[%s, %s] Fitting model with n:' % (train_sl, task), n)
        m.fit(X, Y)
        Pv = m.predict_proba(Xv)[:,1]
        score = roc_auc_score(Yv, Pv)
        print('Fitted with n:', n, 'AUC:', score)
        if score > best_score:
            best_score = score
            best_model = m
            best_n = n

    print('Best n:', best_n)
    return best_model
