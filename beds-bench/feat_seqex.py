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
from absl import app

import tensorflow as tf
from collections import defaultdict
import json


NANO = 10 ** 9
HOURS_24 = 24 * 3600

FILEDIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_JSON = 'config.json'

def config_get(name):
    return json.load(open(os.path.join(FILEDIR, CONFIG_JSON)))[name]

WORKDIR = config_get('work-dir')
DATADIR = os.path.join(WORKDIR, 'data')
SLICE_SETS = list(config_get('partitions').values())

TABLES = ['LABELS', 'PATIENTS', 'ADMISSIONS', 'LABEVENTS', 'CHARTEVENTS', 'INPUTEVENTS', 'OUTPUTEVENTS', 'PRESCRIPTIONS']

np.random.seed(2020)

def val_pids(sl):
    pat_df = pd.read_csv(DATADIR + '/slices/%s/train/PATIENTS.csv' % sl, low_memory=False, parse_dates=['DOB'])
    pids_all = list(pat_df['SUBJECT_ID'])
    np.random.shuffle(pids_all)
    pids_val = set(pids_all[:int(len(pids_all) * 0.2)])
    return pids_val

def load_train_tables(sl, pids_val):
    train_tables, val_tables = {}, {}
    for table in TABLES:
        print('Loading %s.%s ...' % (sl, table))
        full_df = pd.read_csv('%s/slices/%s/train/%s.csv' % (DATADIR, sl, table), low_memory=False)
        train_tables[table] = full_df[~full_df['SUBJECT_ID'].isin(pids_val)]
        val_tables[table] = full_df[full_df['SUBJECT_ID'].isin(pids_val)]
    return train_tables, val_tables

def load_test_tables(SLICES):
    test_tables = {}
    for s in SLICES:
        test_tables[s] = {}
        for t in TABLES:
            test_tables[s][t] = pd.read_csv('%s/slices/%s/test/%s.csv' % (DATADIR, s, t), low_memory=False)
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

def to_date(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df

CODE_TABLES = ['PRESCRIPTIONS', 'INPUTEVENTS', 'OUTPUTEVENTS', 'CHARTEVENTS', 'LABEVENTS']
class PatInfo(object):
    def __init__(self, dob=0, gender="female", in_hospital_expire=0, pid=0, enc=0):
        self.dob = int(dob)
        self.gender = gender
        self.in_hospital_expire = in_hospital_expire
        self.pid = pid
        self.enc = enc
        self.seqs = { c : defaultdict(list) for c in CODE_TABLES }


def process_patients(tables, d):
    pats = to_date(tables['PATIENTS'], ['DOB'])
    for i, row in pats.iterrows():
        pid = str(row['SUBJECT_ID'])
        gender = "female" if row['GENDER'] == 'F' else "male"
        dob = row['DOB'].timestamp()
        if dob < pd.Timestamp(1900, 1, 1).timestamp():
            dob += (300-89) * (365.25 * 24 * 3600)
        d[pid] = PatInfo(dob=dob, pid=pid, gender=gender)
    return d

def process_admissions(tables, d):
    adms = to_date(tables['ADMISSIONS'], ['ADMITTIME'])
    for i, row in adms.iterrows():
        pid = str(row['SUBJECT_ID'])
        pi = d[pid]
        pi.enc = str(row['HADM_ID'])
        pi.timestamp = row['ADMITTIME'].timestamp() + HOURS_24
    return d

def process_ioevents(tables, d, tname):
    inp = to_date(tables[tname], ['CHARTTIME'])
    for i, row in inp.iterrows():
        pid = str(row['SUBJECT_ID'])
        unit = row['AMOUNTUOM']
        val = row['AMOUNT']
        ts = row['CHARTTIME'].timestamp()

        if val != 'ml':
            continue
        pi = d[pid]
        pi.seqs[tname][ts] += val
    return d

def process_codes(tables, d, tname, cname, dname, fname=None):
    labs = to_date(tables[tname], [dname])
    for i, row in labs.iterrows():
        pid = str(row['SUBJECT_ID'])
        code = row[cname]
        if fname:
            flg = row[fname]
            if type(flg) == str:
                code = code + '_' + flg
        ts = row[dname].timestamp()

        pi = d[pid]
        pi.seqs[tname][ts].append(code)

    return d

def process_labels(tables, d):
    for i, row in tables['LABELS'].iterrows():
        pid = str(row['SUBJECT_ID'])

        mort = int(row['HOSPITAL_EXPIRE_FLAG'])
        los3 = int(row['REMAINING_LOS_3'])
        los7 = int(row['REMAINING_LOS_7'])

        pi = d[pid]
        pi.labels_mortality = mort
        pi.labels_los3 = los3
        pi.labels_los7 = los7

    return d

def tables_to_dict(tables):
    d = {}

    process_patients(tables, d)
    process_admissions(tables, d)
    process_ioevents(tables, d, 'INPUTEVENTS')
    process_ioevents(tables, d, 'OUTPUTEVENTS')
    process_codes(tables, d, 'LABEVENTS', 'LOINC_CODE', 'CHARTTIME', 'FLAG')
    process_codes(tables, d, 'CHARTEVENTS', 'ITEMID', 'CHARTTIME')
    process_codes(tables, d, 'PRESCRIPTIONS', 'RXCUI', 'STARTDATE')
    process_labels(tables, d)

    return d

def dict_to_seqex_list(d):
    seqex_list = []
    for pi in d.values():
        seq = tf.train.SequenceExample()
        seq.context.feature['patientId'].bytes_list.value.append(str(pi.pid).encode('utf8'))
        seq.context.feature['currentEncounterId'].int64_list.value.append(int(pi.enc))
        seq.context.feature['Patient.birthDate'].int64_list.value.append(int(pi.dob))
        seq.context.feature['Patient.gender'].bytes_list.value.append(pi.gender.encode('utf8'))
        seq.context.feature['timestamp'].int64_list.value.append(int(pi.timestamp))
        seq.context.feature['label.in_hospital_death.class']
        if pi.labels_mortality:
            seq.context.feature['label.in_hospital_death.class'].bytes_list.value.append('expired'.encode('utf8'))

        seq.context.feature['label.length_of_stay_3plus.class']
        if pi.labels_los3:
            seq.context.feature['label.length_of_stay_3plus.class'].bytes_list.value.append('positive'.encode('utf8'))

        seq.context.feature['label.length_of_stay_7plus.class']
        if pi.labels_los7:
            seq.context.feature['label.length_of_stay_7plus.class'].bytes_list.value.append('positive'.encode('utf8'))

        unique_ts = sorted(set([ts for v in pi.seqs.values() for ts in v.keys()]))
        if len(unique_ts) == 0:
            continue
        seq.context.feature['sequenceLength'].int64_list.value.append(len(unique_ts))
        for ts in unique_ts:
            seq.feature_lists.feature_list['deltaTime'].feature.add().int64_list.value.append(int(pi.timestamp - ts))
            seq.feature_lists.feature_list['eventId'].feature.add().int64_list.value.append(int(ts))
            for ct in CODE_TABLES:
                feat = seq.feature_lists.feature_list[ct].feature.add()
                if ts in pi.seqs[ct]:
                    if ct in ['INPUTEVENTS', 'OUTPUTEVENTS']:
                        val = sum(pi.seqs[ct][ts])
                        feat.float_list.value.append(val)
                    else:
                        for code in pi.seqs[ct][ts]:
                            feat.bytes_list.value.append(str(code).encode('utf8'))
                else:
                    if ct in ['INPUTEVENTS', 'OUTPUTEVENTS']:
                        feat.float_list.value.append(0.)
                        feat.float_list.value.remove(0.)
                    else:
                        feat.bytes_list.value.append(b'')
                        feat.bytes_list.value.remove(b'')
        seqex_list.append(seq)
    return seqex_list


def dump_seqex_list(seqex_list, filename):
    print('Dumping to %s ...' % filename)

    with tf.io.TFRecordWriter(filename) as writer:
        for i, seqex in enumerate(seqex_list):
            patient_id = seqex.context.feature['patientId'].bytes_list.value[0]
            encounter_id = seqex.context.feature['currentEncounterId'].int64_list.value[0]
            seq_length = seqex.context.feature['sequenceLength'].int64_list.value[0]
            #key = 'Pos-%05d/PatientID-%s/%d/EncounterID-%s' % (i, str(patient_id), seq_length, (encounter_id))
            writer.write(seqex.SerializeToString())
    return

def dump_vocab(seqex_list, sl):
    vocab = defaultdict(set)
    for i, seq in enumerate(seqex_list):
        for k, v in seq.context.feature.items():
            for b in v.bytes_list.value:
                vocab[k].add(b)
        for k, vs in seq.feature_lists.feature_list.items():
            for v in vs.feature:
                for b in v.bytes_list.value:
                    vocab[k].add(b)

    for key, vals in vocab.items():
        fd = tf.io.gfile.GFile('%s/seqex/%s/VOCAB/%s.txt' % (DATADIR, sl, key), 'w')
        for val in vals:
            fd.write(val.decode() + '\n')

    fd = tf.io.gfile.GFile('%s/seqex/%s/VOCAB/EmbedConfigFile.pbtxt' % (DATADIR, sl), 'w')
    for k, vals in vocab.items():
        rec = """
features {
  key: "%s"
  value: {
    vocabulary {
      filename: "%s.txt"
      vocabulary_size: %d
    }
    num_oov_buckets: 1
    embedding_dimension: %d
  }
}
"""
        fd.write(rec % (k, k, len(vals), 32))
    return

def dump_seqex(sl, train_tables, val_tables, test_tables):
    for name, tables in [('train', train_tables), ('validation', val_tables), ('test', test_tables)]:
        print('[%s=%s] Creating dict ...' % (name, sl))
        d = tables_to_dict(tables)
        print('[%s=%s] Creating SeqEx list ...' % (name, sl))
        seqex_list = dict_to_seqex_list(d)

        dump_seqex_list(seqex_list, '%s/seqex/%s/%s-00000-of-00001' % (DATADIR, sl, name))
        if name == 'train':
          dump_vocab(seqex_list, sl)

    return

def featurize(sl, SLICES):
    print('SeqExing %s' % sl)

    tf.io.gfile.makedirs('%s/seqex/%s/VOCAB' % (DATADIR, sl))

    pids_val = val_pids(sl)

    train_tables, val_tables = load_train_tables(sl, pids_val)

    test_tables = load_test_tables(SLICES)

    dump_seqex(sl, train_tables, val_tables, test_tables[sl])
    return

def main(args):
    for SLICES in SLICE_SETS:
        for sl in SLICES:
            featurize(sl, SLICES)
            pass

if __name__ == '__main__':
  app.run(main)
