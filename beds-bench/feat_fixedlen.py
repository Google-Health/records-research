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

import pandas as pd
import numpy as np
import sys, os
import csv
import json

FILEDIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_JSON = 'config.json'

def config_get(name):
    return json.load(open(os.path.join(FILEDIR, CONFIG_JSON)))[name]

WORKDIR = config_get('work-dir')
DATADIR = os.path.join(WORKDIR, 'data')
SLICE_SETS = list(config_get('partitions').values())

TABLES = ['LABELS', 'PATIENTS', 'ADMISSIONS', 'LABEVENTS', 'CHARTEVENTS', 'INPUTEVENTS', 'OUTPUTEVENTS', 'PRESCRIPTIONS']

def val_pids(sl):
    csv_file = os.path.join(DATADIR, 'slices', sl, 'train', 'PATIENTS.csv')
    pat_df = pd.read_csv(csv_file, low_memory=False, parse_dates=['DOB'])
    pids_all = list(pat_df['SUBJECT_ID'])
    np.random.shuffle(pids_all)
    pids_val = set(pids_all[:int(len(pids_all) * 0.2)])
    return pids_val

def load_train_tables(sl, pids_val):
    train_tables, val_tables = {}, {}
    for table in TABLES:
        print('Loading %s.%s ...' % (sl, table))
        csv_file = os.path.join(DATADIR, 'slices', sl, 'train', table + '.csv')
        full_df = pd.read_csv(csv_file, low_memory=False)
        train_tables[table] = full_df[~full_df['SUBJECT_ID'].isin(pids_val)]
        val_tables[table] = full_df[full_df['SUBJECT_ID'].isin(pids_val)]
    return train_tables, val_tables

def load_test_tables(SLICES):
    test_tables = {}
    for s in SLICES:
        test_tables[s] = {}
        for t in TABLES:
            csv_file = os.path.join(DATADIR, 'slices', s, 'test', t + '.csv')
            test_tables[s][t] = pd.read_csv(csv_file, low_memory=False)
    return test_tables

def calc_vocab(tables):
    na = set([np.nan, float('nan')])
    print('Calculating LOINC vocab ...')
    lx = tables['LABEVENTS']['LOINC_CODE'].unique()
    print('LOINC vocab size:', len(lx))
    print('Calculating CHARTS vocab ...')
    cx = tables['CHARTEVENTS']['ITEMID'].unique()
    print('CHARTS vocab size:', len(cx))
    print('Calculating RX vocab ...')
    rx = tables['PRESCRIPTIONS']['RXCUI'].unique()
    print('RX vocab size:', len(rx))
    return {'LABEVENTS' : set(lx) - na,
            'CHARTEVENTS' : set(cx) - na,
            'PRESCRIPTIONS' : set(rx) - na}

def vols(table, unit):
    return table.query('AMOUNTUOM == "%s"' % unit)[['SUBJECT_ID', 'AMOUNT']].groupby('SUBJECT_ID').sum()

def to_date(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df

def get_inout(pids, tables):
    print('Loading {IN,OUT}PUTEVENTS into matrix ...')
    df = pd.DataFrame(np.zeros((len(pids), 4)), columns=['INP_ML', 'INP_MG', 'OUT_ML', 'OUT_MG'])
    ix_ml = vols(tables['INPUTEVENTS'], 'ml')
    ix_mg = vols(tables['INPUTEVENTS'], 'mg')
    ox_ml = vols(tables['OUTPUTEVENTS'], 'ml')
    ox_mg = vols(tables['OUTPUTEVENTS'], 'mg')
    df['INP_ML'] = ix_ml['AMOUNT']
    df['INP_MG'] = ix_mg['AMOUNT']
    df['OUT_ML'] = ox_ml['AMOUNT']
    df['OUT_MG'] = ox_mg['AMOUNT']
    return df

def get_demographics(pids, tables):
    print('Loading demographics into matrix ...')
    df = pd.DataFrame(np.zeros((len(pids), 2)), columns=['AGE', 'GENDER'])
    df = df.set_index(pids)
    patients, admissions = tables['PATIENTS'], tables['ADMISSIONS'][['SUBJECT_ID', 'ADMITTIME']]
    join = patients.merge(admissions, on='SUBJECT_ID')
    join = join.set_index('SUBJECT_ID')
    to_date(join, ['DOB', 'ADMITTIME'])
    NANO = 10 ** 9
    SECS_PER_100YEAR = 3600 * 24 * 365.25 * 100
    df['AGE'] = (join['ADMITTIME'].astype(np.int64) / NANO - join['DOB'].astype(np.int64) / NANO) / SECS_PER_100YEAR
    df['AGE'][df['AGE'] > 300] -= (300 - 89) # see https://mimic.physionet.org/mimictables/patients/
    df['GENDER'] = (patients['GENDER'] == 'F')
    return df

def get_labs(pids, tables, vocab):
    print('Loading LABEVENTS into matrix ...')
    loinc = { l : i for i, l in enumerate(vocab['LABEVENTS']) }
    pid = { p : i for i, p in enumerate(pids)}
    data = np.zeros((len(pid), len(loinc)))
    for _, row in tables['LABEVENTS'].iterrows():
        p = row['SUBJECT_ID']
        c = row['LOINC_CODE']
        if c not in loinc:
            continue
        data[pid[p], loinc[c]] += 1
    return pd.DataFrame(data, index=pids, columns=list(loinc.keys()))

def get_rx(pids, tables, vocab):
    print('Loading PRESCRIPTIONS into matrix ...')
    rxcui = { l : i for i, l in enumerate(vocab['PRESCRIPTIONS']) }
    pid = { p : i for i, p in enumerate(pids)}
    data = np.zeros((len(pid), len(rxcui)))
    for _, row in tables['PRESCRIPTIONS'].iterrows():
        p = row['SUBJECT_ID']
        c = row['RXCUI']
        if c not in rxcui:
            continue
        data[pid[p], rxcui[c]] += 1
    return pd.DataFrame(data, index=pids, columns=list(rxcui.keys()))

def get_charts(pids, tables, vocab):
    print('Loading CHARTS into matrix ...')
    itemids = { l : i for i, l in enumerate(vocab['CHARTEVENTS']) }
    pid = { p : i for i, p in enumerate(pids)}
    data = np.zeros((len(pid), len(itemids)))
    for _, row in tables['CHARTEVENTS'].iterrows():
        p = row['SUBJECT_ID']
        c = row['ITEMID']
        if c not in itemids:
            continue
        data[pid[p], itemids[c]] += 1
    return pd.DataFrame(data, index=pids, columns=list(itemids.keys()))

def dump_XY(sl, pfx, tables, vocab):
    # Count of rows
    M = tables['PATIENTS'].shape[0]
    # Count of columns
    # 2 for demographics (age, gender)
    # 4 for Input and Output event totals (in ml, mg)
    # Vocab sizes across rx, lx, cx

    N = 2 + 4 + sum([len(v) for v in vocab.values()])

    pats = tables['PATIENTS'].set_index('SUBJECT_ID')
    pids = pats.index

    demographics = get_demographics(pids, tables)
    inout = get_inout(pids, tables)
    labs = get_labs(pids, tables, vocab)
    cx = get_charts(pids, tables, vocab)
    rx = get_rx(pids, tables, vocab)

    X = demographics.join([cx, inout, labs, rx])
    Y = tables['LABELS']

    csv_x = os.path.join(DATADIR, 'fixedlen', 'train-' + sl, pfx + '_X.csv')
    csv_y = os.path.join(DATADIR, 'fixedlen', 'train-' + sl, pfx + '_Y.csv')
    X.to_csv(csv_x)
    Y.to_csv(csv_y)
    return X, Y


def dump_matrices(sl, train_tables, val_tables, test_tables):
    vocab = calc_vocab(train_tables)
    print('Train=%s' % sl)
    dump_XY(sl, 'train', train_tables, vocab)
    print('Val=%s' % sl)
    dump_XY(sl, 'val', val_tables, vocab)
    for name, tables in test_tables.items():
        print('Train=%s, Test=%s' % (sl, name))
        dump_XY(sl, 'test/%s' % name, tables, vocab)
    return

def featurize(sl, SLICES):
    print('Featurizing %s' % sl)
    os.makedirs('%s/fixedlen/train-%s' % (DATADIR, sl), exist_ok=True)
    os.makedirs('%s/fixedlen/train-%s/test' % (DATADIR, sl), exist_ok=True)

    pids_val = val_pids(sl)
    train_tables, val_tables = load_train_tables(sl, pids_val)

    test_tables = load_test_tables(SLICES)

    dump_matrices(sl, train_tables, val_tables, test_tables)
    return


def main(args):
    for SLICES in SLICE_SETS:
        for sl in SLICES:
            featurize(sl, SLICES)
            pass

if __name__ == '__main__':
    main(sys.argv)
