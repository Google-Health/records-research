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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import os, sys
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
RESDIR = os.path.join(WORKDIR, 'results')
SLICE_NAMES = config_get('slice-desc')

SLICE_TESTS = { 'mimic-adult' : ['mimic-adult', 'mimic-neonate', 'picdb-paed'],
                'mimic-neonate' : ['mimic-neonate', 'mimic-adult', 'picdb-paed'],
                'picdb-paed' : ['picdb-paed', 'mimic-adult', 'mimic-neonate'],
                'mimic-male' : ['mimic-male', 'mimic-female'],
                'mimic-female' : ['mimic-female', 'mimic-male'],
                'mimic-lt50' : ['mimic-lt50', 'mimic-5060', 'mimic-6070', 'mimic-7080', 'mimic-gt80'] }

METRICS = [ ('AUC', 'ABAR', 'blue'),
            ('ECE', 'EBAR', 'brown'),
            ('OOD', 'OBAR', 'red'),
            ('POOD', 'POBAR', 'black')]


# Expected Calibration Error
def ECE(Y, P, n_bins=10):
    #return brier_score_loss(Y, P)
    P, Y = list(Y), list(P)
    l = 1. * len(Y)
    Y_buckets = [ [], [], [], [], [], [], [], [], [], [] ]
    P_buckets = [ [], [], [], [], [], [], [], [], [], [] ]
    for y, p in zip(Y, P):
        idx = int(p * 10)
        if idx == 10:
            idx = 9
        Y_buckets[idx].append(y)
        P_buckets[idx].append(p)

    ece = sum([(len(y_l) / l) *abs(np.mean(y_l) - np.mean(p_l))
               for y_l, p_l in zip(Y_buckets, P_buckets) if len(y_l) > 0])
    return ece

def bootstrap(Y, P, scorefn):
    B = 100
    l = len(Y)
    try:
        Y0 = Y[Y == 0].values
        P0 = P[Y == 0].values
        Y1 = Y[Y == 1].values
        P1 = P[Y == 1].values
    except:
        Y = Y.flatten()
        P = P.flatten()
        Y0 = Y[Y == 0]
        P0 = P[Y == 0]
        Y1 = Y[Y == 1]
        P1 = P[Y == 1]
    l0 = len(Y0)
    l1 = len(Y1)
    choices0 = [np.random.choice(range(l0), l0, replace=True) for _ in range(B)]
    choices1 = [np.random.choice(range(l1), l1, replace=True) for _ in range(B)]
    scores = [scorefn(np.concatenate([Y0[choice0], Y1[choice1]]),
                      np.concatenate([P0[choice0], P1[choice1]]))
              for choice0, choice1 in zip(choices0, choices1)]
    return np.mean(scores), 2 * np.std(scores)

def task_print(task):
    d = {'HOSPITAL_EXPIRE_FLAG' : 'Mortality',
         'REMAINING_LOS_3' : 'LoS 3+',
         'REMAINING_LOS_7' : 'LoS 7+' }
    if task in d:
        return d[task]
    return task

def task_desc(task):
    d = {'Mortality' : 'In Hospital Mortality',
         'LoS 3+' : 'Length of Stay 3+ days',
         'LoS 7+' : 'Length of Stay 7+ days'}
    if task in d:
        return d[task]
    return task

def task_BRNN(task):
    d = {'HOSPITAL_EXPIRE_FLAG' : 'mortality',
         'REMAINING_LOS_3' : 'los3',
         'REMAINING_LOS_7' : 'los7' }
    if task in d:
        return d[task]
    return task

def collect(fd, mname):
    for SLICES in SLICE_SETS:
        for train_sl in SLICES:
            try:
                Y_in = pd.read_csv('%s/train-%s/test/%s_Y.csv' % (FEATDIR, train_sl, train_sl))
                P_in = pd.read_csv('%s/%s/train-%s/test/%s_P.csv' % (PREDDIR, mname, train_sl, train_sl))
            except:
                continue
            Y_in['IO'] = 0
            for task in TASKS:
                for test_sl in SLICES:
                    try:
                        Y = pd.read_csv('%s/train-%s/test/%s_Y.csv' % (FEATDIR, train_sl, test_sl))
                        P = pd.read_csv('%s/%s/train-%s/test/%s_P.csv' % (PREDDIR, mname, train_sl, test_sl))
                    except:
                        continue
                    Y['IO'] = 1
                    auc, abar = bootstrap(Y[task], P[task], roc_auc_score)
                    ece, ebar = bootstrap(Y[task], P[task], ECE)

                    prev = Y[task].mean()
                    def ood_auroc(Y, P):
                        Pfix = P * (1 - prev) / (P + prev - 2 * prev * P)
                        return roc_auc_score(Y, Pfix * (1 - Pfix))

                    Y_io = Y_in.append(Y)
                    P_io = P_in.append(P)
                    ood, obar = bootstrap(Y_io['IO'], P_io[task] * (1 - P_io[task]), roc_auc_score)
                    pood, pobar = bootstrap(Y_io['IO'], P_io[task], ood_auroc)
                    row = '%s,%s,%s,%s,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f' % \
                    (mname, train_sl, test_sl, task_print(task), auc, abar, ece, ebar, ood, obar, pood, pobar)
                    print(row)
                    fd.write(row + '\n')
    pass

def collect_BRNN(fd, basedir, ts='20201210_192808'):
    for SLICES in SLICE_SETS:
        for train_sl in SLICES:
            for task in TASKS:
                try:
                    Y_in = np.load('%s/%s/predictions/train-%s/test-%s/%s/test/labels.npy' % \
                                   (basedir, task_BRNN(task), train_sl, train_sl, ts))
                    L_in = np.load('%s/%s/predictions/train-%s/test-%s/%s/test/logits.npy' % \
                                   (basedir, task_BRNN(task), train_sl, train_sl, ts))
                except:
                    continue

                P_in = 1/(1 + np.exp(-L_in))
                for test_sl in SLICES:
                    try:
                        Y = np.load('%s/%s/predictions/train-%s/test-%s/%s/test/labels.npy' % \
                                    (basedir, task_BRNN(task), train_sl, test_sl, ts))
                        L = np.load('%s/%s/predictions/train-%s/test-%s/%s/test/logits.npy' % \
                                    (basedir, task_BRNN(task), train_sl, test_sl, ts))
                    except:
                        continue
                    P = 1/(1 + np.exp(-L))

                    auc, abar = bootstrap(Y, P, roc_auc_score)
                    ece, ebar = bootstrap(Y, P, ECE)

                    prev = Y.mean()
                    def ood_auroc(Y, P):
                        Pfix = P * (1 - prev) / (P + prev - 2 * prev * P)
                        return roc_auc_score(Y, Pfix * (1 - Pfix))

                    Y_io = np.append(np.zeros(len(Y_in)), np.ones(len(Y)))
                    P_io = np.append(P_in, P)
                    ood, obar = bootstrap(Y_io, P_io * (1 - P_io), roc_auc_score)
                    pood, pobar = bootstrap(Y_io, P_io, ood_auroc)
                    row = '%s,%s,%s,%s,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f' % \
                    ('BRNN', train_sl, test_sl, task_print(task), auc, abar, ece, ebar, ood, obar, pood, pobar)
                    print(row)
                    fd.write(row + '\n')
    pass

def create_results_df():
    brnn_dpath = '/tmp/medical_uncertainty/bayesian_rnn'
    brnn_rpath = '%s/BRNN.csv' % RESDIR
    TS = '20201210_192808'
    if not os.path.exists(brnn_rpath) or os.stat(brnn_rpath).st_size < 100:
        fd = open(brnn_rpath, 'w')
        fd.write('Model,Train,Test,Task,AUC,ABAR,ECE,EBAR,OOD,OBAR,POOD,POBAR\n')
        print('Collecting BRNN...')
        collect_BRNN(fd, brnn_dpath, TS)
        fd.close()
    stats = pd.read_csv(brnn_rpath)
    for mname in MODELS:
        rpath = '%s/%s.csv' % (RESDIR, mname)
        if not os.path.exists(rpath) or os.stat(rpath).st_size < 100:
            fd = open(rpath, 'w')
            fd.write('Model,Train,Test,Task,AUC,ABAR,ECE,EBAR,OOD,OBAR,POOD,POBAR\n')
            print('Collecting %s...' % mname)
            collect(fd, mname)
            fd.close()
        df = pd.read_csv(rpath)
        stats = stats.append(df)
    return stats


def gen_latex(df):
    os.makedirs(os.path.join(RESDIR, 'latex'), exist_ok=True)
    fd = open(os.path.join(RESDIR, 'latex', 'results.tex'), 'w')
    TASKNAMES = df['Task'].unique()

    def P(*args):
        line = ' '.join(str(w) for w in args) + '\n'
        fd.write(line)


    hdr = r'''\documentclass{article}
%%\usepackage[landscape]{geometry}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{fullpage}
\usepackage{diagbox}
\usepackage{multirow}

\begin{document}
\centering
\tiny

'''
    P(hdr)

    for task in TASKNAMES:
        P(r'\section*{', task_desc(task), '}')
        P(r"\begin{tabular}{|l|l|l|" + ''.join([ 'c|' for _ in MODELS ]) + "}")
        P(r"\hline")
        P(r"Train & Test & Metric & " + ' & '.join(MODELS) + r'\\')
        P(r"\hline")

        for sl, SLICES in SLICE_TESTS.items():
            P(r"\multirow{", len(METRICS) * len(SLICES), "}{*}{\\rotatebox{90}{", SLICE_NAMES[sl], "}}")
            for test_sl in SLICES:

                P(r"& \multirow{", len(METRICS), "}{*}{", SLICE_NAMES[test_sl], "}")
                for i, (metric, bar, color) in enumerate(METRICS):
                    if (i > 0):
                        P('&')
                    P(r'& \color{%s}{%s}' % (color, metric))

                    # All stats (across models) to decide which model "wins" the row
                    all_stats = []
                    for model in MODELS:
                        stats = df.query('Train=="%s" and Test=="%s" and Task=="%s" and Model=="%s"' % \
                                         (sl, test_sl, task, model))
                        if stats.shape[0] != 1:
                            continue
                        all_stats.append(stats.iloc[0][metric])

                    for model in MODELS:
                        stats = df.query('Train=="%s" and Test=="%s" and Task=="%s" and Model=="%s"' % \
                                         (sl, test_sl, task, model))
                        if stats.shape[0] != 1:
                            P('& NA')
                        elif sl == test_sl and (metric == 'OOD' or metric == 'POOD'):
                            P('& NA')
                        else:
                            val = stats.iloc[0][metric]
                            valbar = stats.iloc[0][bar]
                            if metric == 'ECE':
                                best = min(all_stats)
                            else:
                                best = max(all_stats)
                            if val != best:
                                P(r'& \color{%s}{%0.3f $\pm$ %0.3f}' % (color, val, valbar))
                            else:
                                P(r'& \textbf{\color{%s}{%0.3f $\pm$ %0.3f}}' % (color, val, valbar))
                    P(r"\\")
                P(r"\cline{2-",len(MODELS)+3,"}")
            P(r"\hline")
            P(r"\hline")
        P(r"\end{tabular}")
    P(r"\end{document}")


def main(args):
    os.makedirs(RESDIR, exist_ok=True)

    df = create_results_df()
    df.to_csv('%s/results.csv' % RESDIR)
    gen_latex(df)

if __name__ == '__main__':
    main(sys.argv)
