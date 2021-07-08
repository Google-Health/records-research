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

from sklearn import preprocessing
import sys, os
import pandas as pd
import numpy as np
import importlib
import json

np.random.seed(2020)

def config_get(name):
    FILEDIR = os.path.dirname(os.path.realpath(__file__))
    CONFIG_JSON = 'config.json'
    return json.load(open(os.path.join(FILEDIR, CONFIG_JSON)))[name]

WORKDIR = config_get('work-dir')
DATADIR = os.path.join(WORKDIR, 'data')
SLICE_SETS = list(config_get('partitions').values())
TASKS = config_get('labels')
PREDDIR = os.path.join(WORKDIR, 'predictions')
FEATDIR = os.path.join(DATADIR, 'fixedlen')
MODELS = config_get('models-fixedlen')

cached_models = {}
def fit_model(mname, train_sl, task, X, Y, Xv, Yv):
    if (mname, train_sl, task) in cached_models:
        print('Returning cached model')
        return cached_models[(train_sl, task)]

    module = importlib.import_module('models.%s' % mname)
    model = module.fit(train_sl, task, X, Y, Xv, Yv)
    cached_models[(train_sl, task)] = model
    return model

def calc_train_size(train_sl):
    Y_train_file = '%s/train-%s/train_Y.csv' % (FEATDIR, train_sl)
    Ys = pd.read_csv(Y_train_file)
    return Ys.shape[0]

def work_slice(mname, train_sl, TEST_SLICES, train_size):
    X_train_file = '%s/train-%s/train_X.csv' % (FEATDIR, train_sl)
    Y_train_file = '%s/train-%s/train_Y.csv' % (FEATDIR, train_sl)
    X_val_file = '%s/train-%s/val_X.csv' % (FEATDIR, train_sl)
    Y_val_file = '%s/train-%s/val_Y.csv' % (FEATDIR, train_sl)

    print('Loading %s ...' % X_train_file)
    X = pd.read_csv(X_train_file, low_memory=False).fillna(0)
    print('Loading %s ...' % X_val_file)
    Xv = pd.read_csv(X_val_file, low_memory=False).fillna(0)
    print('Loading %s ...' % Y_train_file)
    Ys = pd.read_csv(Y_train_file, low_memory=False)
    print('Loading %s ...' % Y_val_file)
    Yvs = pd.read_csv(Y_val_file, low_memory=False)

    total_train_size = X.shape[0]
    if train_size < total_train_size:
        train_idxs = np.random.choice(range(total_train_size), train_size, replace=False)
        X = X.iloc[train_idxs]
        Ys = Ys.iloc[train_idxs]

    print('Scaling X ...')
    scaler = preprocessing.StandardScaler().fit(X)
    for test_sl in TEST_SLICES:
        X_test_file = '%s/train-%s/test/%s_X.csv' % (FEATDIR, train_sl, test_sl)
        print('Loading %s ...' % X_test_file)
        X_test = pd.read_csv(X_test_file, low_memory=False).fillna(0)

        P = pd.DataFrame(np.zeros((X_test.shape[0], len(TASKS))), index=X_test.index, columns=TASKS)
        Pt = pd.DataFrame(np.zeros((X.shape[0], len(TASKS))), index=X.index, columns=TASKS)
        Pv = pd.DataFrame(np.zeros((Xv.shape[0], len(TASKS))), index=Xv.index, columns=TASKS)
        for task in TASKS:
            Y, Yv = Ys[task], Yvs[task]
            m = fit_model(mname, train_sl, task, scaler.transform(X), Y, scaler.transform(Xv), Yv)
            print('Predicting on %s ...' % X_test_file)
            P[task] = m.predict_proba(scaler.transform(X_test))[:,1]
            Pv[task] = m.predict_proba(scaler.transform(Xv))[:,1]
            Pt[task] = m.predict_proba(scaler.transform(X))[:,1]
        os.makedirs('%s/%s/train-%s/test' % (PREDDIR, mname, train_sl), exist_ok=True)
        P.to_csv('%s/%s/train-%s/test/%s_P.csv' % (PREDDIR, mname, train_sl, test_sl))
        Pv.to_csv('%s/%s/train-%s/val_P.csv' % (PREDDIR, mname, train_sl))
        P.to_csv('%s/%s/train-%s/train_P.csv' % (PREDDIR, mname, train_sl))

def work_model(mname):
    print('Working with model: %s' % mname)

    # Make runs reproducible
    np.random.seed(2020)

    for SLICES in SLICE_SETS:
        train_size = min([calc_train_size(sl) for sl in SLICES])
        for sl in SLICES:
            work_slice(mname, sl, SLICES, train_size)

def main(args):
    for mname in MODELS:
        work_model(mname)
        cached_models = {}
        import gc
        gc.collect()

if __name__ == '__main__':
    main(sys.argv)
