#!/usr/bin/env python
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

import sys
import numpy as np
import gzip
from dateutil.parser import parse as date_parse
import datetime
import csv
import random
import gc
from tqdm import tqdm
from collections import defaultdict
import os
import pandas as pd
import json

np.random.seed(2020)
random.seed(2020)


FILEDIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_JSON = 'config.json'

def config_get(name):
    return json.load(open(os.path.join(FILEDIR, CONFIG_JSON)))[name]

WORKDIR = config_get('work-dir')
DATADIR = os.path.join(WORKDIR, 'data')

DEFAULT_MIMIC_DIR = 'mimic-iii-clinical-database-1.4'
DEFAULT_PICDB_DIR = 'paediatric-intensive-care-database-1.0.0'

TOTAL_PATIENTS = 46520 + 12881 # MIMIC + PICDB
TOTAL_ADMISSIONS = 58976 + 13449 # MIMIC + PICDB

# assign patient ID and admission ID randomly across MIMIC and PICDB
PID_REG = random.sample(range(1, 1 + TOTAL_PATIENTS), TOTAL_PATIENTS)
ENC_REG = random.sample(range(1, 1 + TOTAL_ADMISSIONS), TOTAL_ADMISSIONS)

TABLES = ['PATIENTS',
          'ADMISSIONS',
          'LABEVENTS',
          'DIAGNOSES_ICD',
          'PRESCRIPTIONS',
          'INPUTEVENTS',
          'OUTPUTEVENTS',
          'CHARTEVENTS',
          ]

# Forward mappings: (dbname, ID) -> {PID,ENC}_REG
SUB2PID = {}
HAD2ENC = {}

def sub2pid(sub, dbname):
    if (sub, dbname) in SUB2PID:
        return SUB2PID[(sub, dbname)]
    pid = PID_REG.pop()
    SUB2PID[(sub, dbname)] = pid
    return pid

def had2enc(had, dbname):
    if (had, dbname) in HAD2ENC:
        return HAD2ENC[(had, dbname)]
    enc = ENC_REG.pop()
    HAD2ENC[(had, dbname)] = enc
    return enc

def CSVReader(dirname, tname, cols=None):
    try:
        fd = open('%s/%s.csv' % (dirname, tname))
    except:
        fd = gzip.open('%s/%s.csv.gz' % (dirname, tname), 'rt')
    return csv.DictReader(fd)

########### PATIENTS table ##############

COL_PATIENTS = ['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']
ALL_PATIENTS = []
def HANDLE_PATIENTS(dirname, tname, dbname, size):
    for rec in tqdm(CSVReader(dirname, tname), total=size):
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        sex = rec['GENDER']
        #dob = date_parse(rec['DOB'])
        dob = rec['DOB']
        exp = int(rec['EXPIRE_FLAG'])
        #dod = None if exp == 0 else date_parse(rec['DOD'])
        dod = rec['DOD']

        ALL_PATIENTS.append((pid, sex, dob, dod, exp))
    return

MIMIC_PATIENTS = HANDLE_PATIENTS
PICDB_PATIENTS = HANDLE_PATIENTS

########### ADMISSIONS table ##############

COL_ADMISSIONS = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'ETHNICITY', 'HOSPITAL_EXPIRE_FLAG']
ALL_ADMISSIONS = []
def HANDLE_ADMISSIONS(dirname, tname, dbname, size):
    def picdb_ethnic(eth):
        if eth == 'Other':
            return 'OTHER'
        return 'ASIAN - CHINESE'

    for rec in tqdm(CSVReader(dirname, tname), total=size):
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname)
        #adm = date_parse(rec['ADMITTIME'])
        adm = rec['ADMITTIME']
        #dis = date_parse(rec['DISCHTIME'])
        dis = rec['DISCHTIME']
        eth = rec['ETHNICITY'] if dbname == 'MIMIC' else picdb_ethnic(rec['ETHNICITY'])
        exp = int(rec['HOSPITAL_EXPIRE_FLAG'])

        ALL_ADMISSIONS.append((pid, enc, adm, dis, eth, exp))
    return

MIMIC_ADMISSIONS = HANDLE_ADMISSIONS
PICDB_ADMISSIONS = HANDLE_ADMISSIONS

########### CHARTEVENTS table ##############

(TEMPERATURE,
 HEARTRATE,
 RESPIRATORYRATE,
 OXYGENSATURATION,
 GLUCOSE,
 HEIGHT,
 WEIGHT,
 DIASTOLICPRESSURE,
 SYSTOLICPRESSURE) = range(9)

flip_dict = lambda d: { v : k for k, vals in d.items() for v in vals }

# Obtained from MIMIC-Extract paper's Appendix
MIMIC_CHART_ITEMS = flip_dict({ TEMPERATURE       : [223761, 677, 676, 678, 679, 223762],  # Celcius
                                HEARTRATE         : [211, 220045], # BPM
                                RESPIRATORYRATE   : [224422, 618, 220210, 224689, 614, 651, 224690, 615],
                                OXYGENSATURATION  : [646, 50817, 834, 220277, 220227], # [0-100] (percentage)
                                GLUCOSE           : [220621, 225664, 811, 807, 226537, 1529],
                                HEIGHT            : [226707, 226730, 1394],
                                WEIGHT            : [226531, 763, 224639, 226512],
                                DIASTOLICPRESSURE : [224643, 225310, 220180, 8555, 220051, 8368, 8441, 8440],
                                SYSTOLICPRESSURE  : [442, 227243, 224167, 220179, 225309, 6701, 220050, 51, 455] })

# Obtained from D_ITEMS table in PICDB
PICDB_CHART_ITEMS = { 1001 : TEMPERATURE,
                      1002 : HEARTRATE, # actually Pulse in PICDB
                      1003 : HEARTRATE,
                      1004 : RESPIRATORYRATE,
                      1006 : OXYGENSATURATION,
                      1011 : GLUCOSE,
                      1013 : HEIGHT,
                      1014 : WEIGHT,
                      1015 : DIASTOLICPRESSURE,
                      1016 : SYSTOLICPRESSURE }

# TODO: 1009 and 1010 to load into INPUTEVENTS and OUTPUTEVENTS resp.

COL_CHARTEVENTS = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'STORETIME', 'VALUE', 'VALUEUOM']
ALL_CHARTEVENTS = []
def HANDLE_CHARTEVENTS(dirname, tname, dbname, size):
    items = PICDB_CHART_ITEMS if dbname == 'PICDB' else MIMIC_CHART_ITEMS

    for rec in tqdm(CSVReader(dirname, tname), total=size):
        itm_raw = int(rec['ITEMID'])
        if not itm_raw in items:
            continue
        itm = items[itm_raw]
        if rec['VALUENUM'] == '':
            continue
        val = float(rec['VALUENUM'])
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname)
        #ctm = date_parse(rec['CHARTTIME'])
        ctm = rec['CHARTTIME']
        #stm = date_parse(rec['STORETIME']) if rec['STORETIME'] != '' else None
        stm = rec['STORETIME']
        uom = rec['VALUEUOM']

        ALL_CHARTEVENTS.append((pid, enc, itm, ctm, stm, val, uom))
    return

MIMIC_CHARTEVENTS = HANDLE_CHARTEVENTS
PICDB_CHARTEVENTS = HANDLE_CHARTEVENTS

########### DIAGNOSES_ICD table ##############

COL_DIAGNOSES_ICD = ['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']
ALL_DIAGNOSES_ICD = []
def MIMIC_DIAGNOSES_ICD(dirname, tname, dbname, size):
    for rec in tqdm(CSVReader(dirname, tname), total=size):
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname)
        seq = int(rec['SEQ_NUM']) if rec['SEQ_NUM'] != '' else None
        if not seq:
            continue
        icd = rec['ICD9_CODE']

        ALL_DIAGNOSES_ICD.append((pid, enc, seq, icd))
    return

def PICDB_DIAGNOSES_ICD(dirname, tname, dbname, size):
    cn_en = {}
    for rec in CSVReader(dirname, 'D_ICD_DIAGNOSES'):
        cn = rec['ICD10_CODE_CN']
        en = rec['ICD10_CODE'].replace('.', '')
        cn_en[cn] = en

    icd10_9 = {}
    for line in open('%s/res/2018_I10gem.txt' % FILEDIR):
        icd10, icd9, flags = line.split()
        if icd10 in icd10_9:
            continue
        icd10_9[icd10] = icd9

    for line in open('%s/res/icd9_10.txt' % FILEDIR):
        icd10, icd9s = line.strip().split(':')
        icd9s = icd9s.split(',')
        if icd10 in icd10_9:
            continue
        for icd9 in icd9s:
            icd10_9[icd10] = icd9

    for line in open('%s/res/icd910cn.txt' % FILEDIR):
        _, icd9, icd10 = line.strip().split(',')
        icd10 = icd10.replace('.', '')
        icd9 = icd9.replace('.', '')
        if icd10 in icd10_9:
            continue
        icd10_9[icd10] = icd9

    def icd10cn_to_icd9(icd10cn):
        icd10 = cn_en[icd10cn]
        if icd10 in icd10_9:
            return icd10_9[icd10]
        for k, v in icd10_9.items():
            if k.startswith(icd10) or icd10.startswith(k):
                icd10_9[icd10] = v
                return v

    unmapped = set()
    for rec in tqdm(CSVReader(dirname, tname), total=size):
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname)
        seq = int(rec['SEQ_NUM'])
        icd = icd10cn_to_icd9(rec['ICD10_CODE_CN'])
        if not icd:
            unmapped.add(rec['ICD10_CODE_CN'])
            continue

        ALL_DIAGNOSES_ICD.append((pid, enc, seq, icd))

    if len(unmapped) > 0:
        print('Unampped ICD10:', sorted(unmapped))
    return


########### LABEVENTS table ##############

COL_LABEVENTS = ['SUBJECT_ID', 'HADM_ID', 'LOINC_CODE', 'CHARTTIME',
                 'VALUE', 'VALUEUOM', 'FLAG']
ALL_LABEVENTS = []
def HANDLE_LABEVENTS(dirname, tname, dbname, size):
    item_loinc = {}
    for rec in CSVReader(dirname, 'D_LABITEMS'):
        item = int(rec['ITEMID'])
        loinc = rec['LOINC_CODE']
        if loinc and len(loinc) > 2:
            item_loinc[item] = loinc

    for rec in tqdm(CSVReader(dirname, tname), total=size):
        item = int(rec['ITEMID'])
        if item not in item_loinc:
            continue
        loi = item_loinc[item]
        flg = rec['FLAG']
        if rec['FLAG'] != '' and dbname == 'PICDB':
            flg = '' if rec['FLAG'] == 'z' else 'abnormal'
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname) if rec['HADM_ID'] != '' else None
        ctm = rec['CHARTTIME']
        if rec['VALUENUM'] == '':
            continue
        val = float(rec['VALUENUM'])
        uom = rec['VALUEUOM']

        ALL_LABEVENTS.append((pid, enc, loi, ctm, val, uom, flg))
    pass

MIMIC_LABEVENTS = HANDLE_LABEVENTS
PICDB_LABEVENTS = HANDLE_LABEVENTS


########### PRESCRIPTIONS table ##############

COL_PRESCRIPTIONS = ['SUBJECT_ID', 'HADM_ID', 'STARTDATE', 'ENDDATE', 'RXCUI',
                     'DOSE_VAL_RX', 'DOSE_UNIT_RX']
ALL_PRESCRIPTIONS = []
def MIMIC_PRESCRIPTIONS(dirname, tname, dbname, size):
    ndc_rxcui = defaultdict(set)
    for rec in csv.DictReader(open('%s/res/ndcrxcui.csv' % FILEDIR)):
        ndc = rec['NDC']
        rxcui = rec['RXCUI']
        ndc_rxcui[ndc].add(rxcui)

    for rec in tqdm(CSVReader(dirname, tname), total=size):
        ndc = rec['NDC']
        if ndc == '' or ndc == '0':
            continue
        rxcuis = ndc_rxcui[ndc]
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname) if rec['HADM_ID'] != '' else None
        std = rec['STARTDATE']
        end = rec['ENDDATE']
        val = rec['DOSE_VAL_RX']
        unt = rec['DOSE_UNIT_RX']

        for rxcui in rxcuis:
            ALL_PRESCRIPTIONS.append((pid, enc, std, end, rxcui, val, unt))
    return

def PICDB_PRESCRIPTIONS(dirname, tname, dbname, size):
    name_rxcui = defaultdict(set)
    for line in open('%s/res/medex-output.txt' % FILEDIR):
        idx, out = line.split('\t')
        words = out.split('|')
        name, rxcui, rxcui_gen = words[0].replace('.', ''), words[11], words[12]
        if rxcui != '':
            name_rxcui[name].add(rxcui)
        if rxcui_gen != '':
            name_rxcui[name].add(rxcui_gen)

    for rec in tqdm(CSVReader(dirname, tname), total=size):
        name = rec['DRUG_NAME_EN']
        if name not in name_rxcui:
            # TBD record missing codes and counts
            continue
        rxcuis = name_rxcui[name]
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname) if rec['HADM_ID'] != '' else None
        std = rec['STARTDATE']
        end = rec['ENDDATE']
        val = rec['DOSE_VAL_RX']
        unt = rec['DOSE_UNIT_RX']

        for rxcui in rxcuis:
            ALL_PRESCRIPTIONS.append((pid, enc, std, end, rxcui, val, unt))
    return

########### INPUTEVENTS table ##############

# To append the generators for INPUTEVENTS_CV and INPUTEVENTS_MV
def concat(*iterables):
    for iterable in iterables:
        yield from iterable
    return

COL_INPUTEVENTS = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'AMOUNT', 'AMOUNTUOM']
ALL_INPUTEVENTS = []
def MIMIC_INPUTEVENTS(dirname, tname, dbname, size):
    rdr = concat(CSVReader(dirname, tname + '_MV'),
                 CSVReader(dirname, tname + '_CV'))

    for rec in tqdm(rdr, total=size):
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname) if rec['HADM_ID'] != '' else None
        ctm = rec['CHARTTIME'] if 'CHARTTIME' in rec else rec['ENDTIME']
        if rec['AMOUNT'] == '':
            continue
        amt = float(rec['AMOUNT'])
        uom = rec['AMOUNTUOM']

        ALL_INPUTEVENTS.append((pid, enc, ctm, amt, uom))
    return

def PICDB_INPUTEVENTS(dirname, tname, dbname, size):
    for rec in tqdm(CSVReader(dirname, tname), total=size):
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname) if rec['HADM_ID'] != '' else None
        ctm = rec['CHARTTIME']
        if rec['AMOUNT'] == '':
            continue
        amt = float(rec['AMOUNT'])
        uom = rec['AMOUNTUOM']

        ALL_INPUTEVENTS.append((pid, enc, ctm, amt, uom))
    return


########### OUTPUTEVENTS table ##############

COL_OUTPUTEVENTS = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'AMOUNT', 'AMOUNTUOM']
ALL_OUTPUTEVENTS = []
def HANDLE_OUTPUTEVENTS(dirname, tname, dbname, size):
    for rec in tqdm(CSVReader(dirname, tname), total=size):
        pid = sub2pid(int(rec['SUBJECT_ID']), dbname)
        enc = had2enc(int(rec['HADM_ID']), dbname) if rec['HADM_ID'] != '' else None
        ctm = rec['CHARTTIME']
        if rec['VALUE'] == '':
            continue
        amt = float(rec['VALUE'])
        uom = rec['VALUEUOM']

        ALL_OUTPUTEVENTS.append((pid, enc, ctm, amt, uom))
    return
MIMIC_OUTPUTEVENTS = HANDLE_OUTPUTEVENTS
PICDB_OUTPUTEVENTS = HANDLE_OUTPUTEVENTS

# for TQDM
TABLE_SIZE = {
    'MIMIC': { 'PATIENTS'      : 46520,
               'ADMISSIONS'    : 58976,
               'LABEVENTS'     : 27854055,
               'CHARTEVENTS'   : 330712483,
               'DIAGNOSES_ICD' : 651047,
               'PRESCRIPTIONS' : 4156450,
               'INPUTEVENTS'   : 21146926,
               'OUTPUTEVENTS'  : 4349218
              },
    'PICDB': { 'PATIENTS'      : 12881,
               'ADMISSIONS'    : 13449,
               'CHARTEVENTS'   : 2278978,
               'LABEVENTS'     : 10094117,
               'DIAGNOSES_ICD' : 13365,
               'PRESCRIPTIONS' : 1256591,
               'INPUTEVENTS'   : 26884,
               'OUTPUTEVENTS'  : 39891
              }}

def dump_csv(table, cols, rows):
    wrt = csv.writer(open('%s/full/%s.csv' % (DATADIR, table), 'w'), quotechar='"')
    wrt.writerow(cols)
    for row in rows:
        wrt.writerow(row)

def dump_metadata():
    METADATA = {}
    for (sub, dbname), pid in SUB2PID.items():
        METADATA[pid] = { 'DB' : dbname,
                          'SUBJECT_ID' : sub }
    print('Dumping METADATA ...')
    DOB = {}
    for row in ALL_PATIENTS:
        gender_idx = COL_PATIENTS.index('GENDER')
        subject_idx = COL_PATIENTS.index('SUBJECT_ID')
        dob_idx = COL_PATIENTS.index('DOB')
        METADATA[pid]['GENDER'] = row[gender_idx]
        pid = row[subject_idx]
        dob = date_parse(row[dob_idx])
        DOB[pid] = dob

    for row in ALL_ADMISSIONS:
        adtime_idx = COL_ADMISSIONS.index('ADMITTIME')
        dstime_idx = COL_ADMISSIONS.index('DISCHTIME')
        subject_idx = COL_ADMISSIONS.index('SUBJECT_ID')
        ethnicity_idx = COL_ADMISSIONS.index('ETHNICITY')
        expire_idx = COL_ADMISSIONS.index('HOSPITAL_EXPIRE_FLAG')
        adtime = date_parse(row[adtime_idx])
        dstime = date_parse(row[dstime_idx])
        los_timedelta = dstime - adtime

        if los_timedelta < datetime.timedelta(days=1, hours=6):
            # Too short gap time for the label not to leak info
            continue
        remain_los_3 = los_timedelta > datetime.timedelta(days=3+1)
        remain_los_7 = los_timedelta > datetime.timedelta(days=7+1)

        pid = row[subject_idx]
        eth = row[ethnicity_idx]
        dob = DOB[pid]
        age_days = (adtime - dob).days
        replace_eth = ['UNKNOWN/NOT SPECIFIED', 'PATIENT DECLINED TO ANSWER', 'OTHER', 'UNABLE TO OBTAIN']
        md_pid = METADATA[pid]
        #if 'ETHNICITY' in md_pid and md_pid['ETHNICITY'] in replace_eth and eth not in replace_eth:
        #    print('Replacing %s with %s' % (md_pid['ETHNICITY'], eth))
        md_pid['HOSPITAL_EXPIRE_FLAG'] = row[expire_idx]
        md_pid['REMAINING_LOS_3'] = remain_los_3
        md_pid['REMAINING_LOS_7'] = remain_los_7
        if not 'ETHNICITY' in md_pid or md_pid['ETHNICITY'] in replace_eth:
            md_pid['ETHNICITY'] = eth
        if not 'AGE_IN_DAYS' in md_pid or md_pid['AGE_IN_DAYS'] > age_days:
            md_pid['AGE_IN_DAYS'] = age_days
            md_pid['DATA_START'] = str(adtime)
            md_pid['DATA_END'] = str(adtime + datetime.timedelta(days=1))

    csvw = csv.writer(open('%s/METADATA.csv' % DATADIR, 'w'))
    COLS = ['SUBJECT_ID'] + list(md_pid.keys())
    csvw.writerow(COLS)

    # Only those patients having an admission that was sufficiently long (> 30hrs)
    METADATA = { pid : md_pid for pid, md_pid in METADATA.items() if 'AGE_IN_DAYS' in md_pid }

    for pid, md_pid in METADATA.items():
        row = [pid] + list(md_pid.values())
        csvw.writerow(row)

    dw = csv.writer(open('%s/full/D_ITEMS.csv' % DATADIR, 'w'))
    dw.writerow(['ITEMID', 'DESCRIPTION'])
    for item in ['TEMPERATURE', 'HEARTRATE', 'RESPIRATORYRATE', 'OXYGENSATURATION',
                 'GLUCOSE', 'HEIGHT', 'WEIGHT', 'DIASTOLICPRESSURE',
                 'SYSTOLICPRESSURE']:
        dw.writerow([eval(item), item])

    return METADATA

def extract_table(tname, slice_name, split_name, dates, ALL_ROWS, ALL_COLS, COLS, date_col):
    os.makedirs('%s/slices/%s/%s' % (DATADIR, slice_name, split_name), exist_ok=True)
    fd = open('%s/slices/%s/%s/%s.csv' % (DATADIR, slice_name, split_name, tname), 'w')
    tw = csv.writer(fd)
    idxs = [ALL_COLS.index(name) for name in COLS] if COLS else None
    pid_idx = ALL_COLS.index('SUBJECT_ID')
    date_idx = ALL_COLS.index(date_col) if date_col else None
    EXT_ROWS = []
    print('Dumping to %s.%s (%s) ...' % (slice_name, tname, split_name))
    tw.writerow(COLS if COLS else ALL_COLS)
    for r, row in enumerate(ALL_ROWS):
        pid = row[pid_idx]
        if pid not in dates:
            continue
        if date_idx:
            start, end = dates[pid]
            if row[date_idx] == '':
                continue
            date = date_parse(row[date_idx])
            if date < start or date > end:
                continue
        if idxs:
            tw.writerow([row[i] for i in idxs])
        else:
            tw.writerow(row)
    return

def extract_labels(slice_name, split_name, labels):
    os.makedirs('%s/slices/%s/%s' % (DATADIR, slice_name, split_name), exist_ok=True)
    lw = csv.writer(open('%s/slices/%s/%s/LABELS.csv' % (DATADIR, slice_name, split_name), 'w'))
    lw.writerow(['SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG', 'REMAINING_LOS_3', 'REMAINING_LOS_7'])
    for pid, (exp, los3, los7) in labels.items():
        lw.writerow([pid, exp, los3, los7])
    return

def train_test_split(dates, labels, p=0.8):
    pid_0 = [ pid for pid, row in labels.items() if row[0] == 0 ]
    pid_1 = [ pid for pid, row in labels.items() if row[0] == 1 ]
    np.random.shuffle(pid_0)
    np.random.shuffle(pid_1)
    train_pids = set( pid_0[:int(len(pid_0) * p)] + pid_1[:int(len(pid_1) * p)] )
    test_pids  = set( pid_0[int(len(pid_0) * p):] + pid_1[int(len(pid_1) * p):] )
    train_dates = { pid : row for pid, row in dates.items() if pid in train_pids }
    train_labels = { pid : row for pid, row in labels.items() if pid in train_pids }
    test_dates = { pid : row for pid, row in dates.items() if pid in test_pids }
    test_labels = { pid : row for pid, row in labels.items() if pid in test_pids }
    return train_dates, train_labels, test_dates, test_labels

def create_slice(MD, slice_name, condition):
    slice = MD.query(condition)
    all_dates = { pid : (date_parse(row['DATA_START']), date_parse(row['DATA_END']))
                  for pid, row in slice.iterrows() }
    all_labels = { pid : (int(row['HOSPITAL_EXPIRE_FLAG']), int(row['REMAINING_LOS_3']),
                          int(row['REMAINING_LOS_7'])) for pid, row in slice.iterrows() }
    print('Creating slice "%s" ...' % slice_name)

    train_dates, train_labels, test_dates, test_labels = train_test_split(all_dates, all_labels)

    for (split_name, labels, dates) in [('train', train_labels, train_dates),
                                        ('test', test_labels, test_dates)]:
        extract_labels(slice_name, split_name, labels)
        extract_table('PATIENTS', slice_name, split_name, dates,
                      ALL_PATIENTS, COL_PATIENTS,
                      ['SUBJECT_ID', 'GENDER', 'DOB'], None)
        extract_table('ADMISSIONS', slice_name, split_name, dates,
                      ALL_ADMISSIONS, COL_ADMISSIONS,
                      ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ETHNICITY'],
                      'ADMITTIME')
        extract_table('CHARTEVENTS', slice_name, split_name, dates,
                      ALL_CHARTEVENTS, COL_CHARTEVENTS, COL_CHARTEVENTS, 'CHARTTIME')
        extract_table('LABEVENTS', slice_name, split_name, dates,
                      ALL_LABEVENTS, COL_LABEVENTS, None, 'CHARTTIME')
        extract_table('INPUTEVENTS', slice_name, split_name, dates,
                      ALL_INPUTEVENTS, COL_INPUTEVENTS, None, 'CHARTTIME')
        extract_table('OUTPUTEVENTS', slice_name, split_name, dates,
                  ALL_OUTPUTEVENTS, COL_OUTPUTEVENTS, None, 'CHARTTIME')
        extract_table('PRESCRIPTIONS', slice_name, split_name, dates,
                      ALL_PRESCRIPTIONS, COL_PRESCRIPTIONS, None, 'STARTDATE')
    # TODO: add D_ITEMS into slice
    return

def create_slices(METADATA):
    MD = pd.DataFrame.from_dict(METADATA, orient='index')
    sliceinfo = config_get('partition-slices')
    for slice_name, condition in sliceinfo.items():
        create_slice(MD, slice_name, condition)
    pass

def main(args):
    bmark_full = '%s/full' % DATADIR
    if not os.path.isdir(bmark_full):
        os.makedirs(bmark_full, exist_ok=True)

    for table in TABLES:
        for dbname in ['MIMIC', 'PICDB']:
            dirname = '%s/%s' % (WORKDIR, eval('DEFAULT_' + dbname + '_DIR'))
            fn = eval(dbname + '_' + table)
            print('Loading %s.%s ...' % (dbname, table))
            fn(dirname, table, dbname, TABLE_SIZE[dbname][table])

        cols = eval('COL_' + table)
        rows = eval('ALL_' + table)
        print('Dumping CSV for %s' % table)
        dump_csv(table, cols, rows)

    assert len(PID_REG) == 0
    assert len(ENC_REG) == 0

    METADATA = dump_metadata()

    create_slices(METADATA)

    return METADATA

if __name__ == '__main__':
    main(sys.argv)
