"""Copyright 2019 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
from __future__ import print_function

import cPickle as pickle
import csv
import os
import sys
import numpy as np
import sklearn.model_selection as ms
import tensorflow as tf


class EncounterInfo(object):

  def __init__(self, patient_id, encounter_id, encounter_timestamp, expired,
               readmission):
    self.patient_id = patient_id
    self.encounter_id = encounter_id
    self.encounter_timestamp = encounter_timestamp
    self.expired = expired
    self.readmission = readmission
    self.dx_ids = []
    self.rx_ids = []
    self.labs = {}
    self.physicals = []
    self.treatments = []


def process_patient(infile, encounter_dict, hour_threshold=24):
  inff = open(infile, 'r')
  count = 0
  patient_dict = {}
  for line in csv.DictReader(inff):
    if count % 10000 == 0:
      sys.stdout.write('%d\r' % count)
      sys.stdout.flush()

    patient_id = line['patienthealthsystemstayid']
    encounter_id = line['patientunitstayid']
    encounter_timestamp = -int(line['hospitaladmitoffset'])
    if patient_id not in patient_dict:
      patient_dict[patient_id] = []
    patient_dict[patient_id].append((encounter_timestamp, encounter_id))
  inff.close()
  print('')

  patient_dict_sorted = {}
  for patient_id, time_enc_tuples in patient_dict.iteritems():
    patient_dict_sorted[patient_id] = sorted(time_enc_tuples)

  enc_readmission_dict = {}
  for patient_id, time_enc_tuples in patient_dict_sorted.iteritems():
    for time_enc_tuple in time_enc_tuples[:-1]:
      enc_id = time_enc_tuple[1]
      enc_readmission_dict[enc_id] = True
    last_enc_id = time_enc_tuples[-1][1]
    enc_readmission_dict[last_enc_id] = False

  inff = open(infile, 'r')
  count = 0
  for line in csv.DictReader(inff):
    if count % 10000 == 0:
      sys.stdout.write('%d\r' % count)
      sys.stdout.flush()

    patient_id = line['patienthealthsystemstayid']
    encounter_id = line['patientunitstayid']
    encounter_timestamp = -int(line['hospitaladmitoffset'])
    discharge_status = line['unitdischargestatus']
    duration_minute = float(line['unitdischargeoffset'])
    expired = True if discharge_status == 'Expired' else False
    readmission = enc_readmission_dict[encounter_id]

    if duration_minute > 60. * hour_threshold:
      continue

    ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired,
                       readmission)
    if encounter_id in encounter_dict:
      print('Duplicate encounter ID!!')
      sys.exit(0)
    encounter_dict[encounter_id] = ei
    count += 1

  inff.close()
  print('')

  return encounter_dict


def process_admission_dx(infile, encounter_dict):
  inff = open(infile, 'r')
  count = 0
  missing_eid = 0
  for line in csv.DictReader(inff):
    if count % 10000 == 0:
      sys.stdout.write('%d\r' % count)
      sys.stdout.flush()

    encounter_id = line['patientunitstayid']
    dx_id = line['admitdxpath'].lower()

    if encounter_id not in encounter_dict:
      missing_eid += 1
      continue
    encounter_dict[encounter_id].dx_ids.append(dx_id)
    count += 1
  inff.close()
  print('')
  print('Admission Diagnosis without Encounter ID: %d' % missing_eid)

  return encounter_dict


def process_diagnosis(infile, encounter_dict):
  inff = open(infile, 'r')
  count = 0
  missing_eid = 0
  for line in csv.DictReader(inff):
    if count % 10000 == 0:
      sys.stdout.write('%d\r' % count)
      sys.stdout.flush()

    encounter_id = line['patientunitstayid']
    dx_id = line['diagnosisstring'].lower()

    if encounter_id not in encounter_dict:
      missing_eid += 1
      continue
    encounter_dict[encounter_id].dx_ids.append(dx_id)
    count += 1
  inff.close()
  print('')
  print('Diagnosis without Encounter ID: %d' % missing_eid)

  return encounter_dict


def process_treatment(infile, encounter_dict):
  inff = open(infile, 'r')
  count = 0
  missing_eid = 0

  for line in csv.DictReader(inff):
    if count % 10000 == 0:
      sys.stdout.write('%d\r' % count)
      sys.stdout.flush()

    encounter_id = line['patientunitstayid']
    treatment_id = line['treatmentstring'].lower()

    if encounter_id not in encounter_dict:
      missing_eid += 1
      continue
    encounter_dict[encounter_id].treatments.append(treatment_id)
    count += 1
  inff.close()
  print('')
  print('Treatment without Encounter ID: %d' % missing_eid)
  print('Accepted treatments: %d' % count)

  return encounter_dict


def build_seqex(enc_dict,
                skip_duplicate=False,
                min_num_codes=1,
                max_num_codes=50):
  key_list = []
  seqex_list = []
  num_cut = 0
  num_duplicate = 0
  count = 0
  num_dx_ids = 0
  num_treatments = 0
  num_unique_dx_ids = 0
  num_unique_treatments = 0
  min_dx_cut = 0
  min_treatment_cut = 0
  max_dx_cut = 0
  max_treatment_cut = 0
  num_expired = 0
  num_readmission = 0

  for _, enc in enc_dict.iteritems():
    if skip_duplicate:
      if (len(enc.dx_ids) > len(set(enc.dx_ids)) or
          len(enc.treatments) > len(set(enc.treatments))):
        num_duplicate += 1
        continue

    if len(set(enc.dx_ids)) < min_num_codes:
      min_dx_cut += 1
      continue

    if len(set(enc.treatments)) < min_num_codes:
      min_treatment_cut += 1
      continue

    if len(set(enc.dx_ids)) > max_num_codes:
      max_dx_cut += 1
      continue

    if len(set(enc.treatments)) > max_num_codes:
      max_treatment_cut += 1
      continue

    count += 1
    num_dx_ids += len(enc.dx_ids)
    num_treatments += len(enc.treatments)
    num_unique_dx_ids += len(set(enc.dx_ids))
    num_unique_treatments += len(set(enc.treatments))

    seqex = tf.train.SequenceExample()
    seqex.context.feature['patientId'].bytes_list.value.append(enc.patient_id +
                                                               ':' +
                                                               enc.encounter_id)
    if enc.expired:
      seqex.context.feature['label.expired'].bytes_list.value.append('expired')
      num_expired += 1
    if enc.readmission:
      seqex.context.feature['label.readmission'].bytes_list.value.append(
          'readmission')
      num_readmission += 1

    dx_ids = seqex.feature_lists.feature_list['dx_ids']
    dx_ids.feature.add().bytes_list.value.extend(list(set(enc.dx_ids)))

    proc_ids = seqex.feature_lists.feature_list['proc_ids']
    proc_ids.feature.add().bytes_list.value.extend(list(set(enc.treatments)))

    seqex_list.append(seqex)
    key = seqex.context.feature['patientId'].bytes_list.value[0]
    key_list.append(key)

  print('Filtered encounters due to duplicate codes: %d' % num_duplicate)
  print('Filtered encounters due to thresholding: %d' % num_cut)
  print('Average num_dx_ids: %f' % (num_dx_ids / count))
  print('Average num_treatments: %f' % (num_treatments / count))
  print('Average num_unique_dx_ids: %f' % (num_unique_dx_ids / count))
  print('Average num_unique_treatments: %f' % (num_unique_treatments / count))
  print('Min dx cut: %d' % min_dx_cut)
  print('Min treatment cut: %d' % min_treatment_cut)
  print('Max dx cut: %d' % max_dx_cut)
  print('Max treatment cut: %d' % max_treatment_cut)
  print('Number of expired: %d' % num_expired)
  print('Number of readmission: %d' % num_readmission)

  return key_list, seqex_list


def select_train_valid_test(key_list, random_seed=1234):
  key_train, key_temp = ms.train_test_split(
      key_list, test_size=0.2, random_state=random_seed)
  key_valid, key_test = ms.train_test_split(
      key_temp, test_size=0.5, random_state=random_seed)
  return key_train, key_valid, key_test


def count_conditional_prob_dp(seqex_list, output_path, train_key_set=None):
  dx_freqs = {}
  proc_freqs = {}
  dp_freqs = {}
  total_visit = 0
  for seqex in seqex_list:
    if total_visit % 1000 == 0:
      sys.stdout.write('Visit count: %d\r' % total_visit)
      sys.stdout.flush()

    key = seqex.context.feature['patientId'].bytes_list.value[0]
    if (train_key_set is not None and key not in train_key_set):
      total_visit += 1
      continue

    dx_ids = seqex.feature_lists.feature_list['dx_ids'].feature[
        0].bytes_list.value
    proc_ids = seqex.feature_lists.feature_list['proc_ids'].feature[
        0].bytes_list.value

    for dx in dx_ids:
      if dx not in dx_freqs:
        dx_freqs[dx] = 0
      dx_freqs[dx] += 1

    for proc in proc_ids:
      if proc not in proc_freqs:
        proc_freqs[proc] = 0
      proc_freqs[proc] += 1

    for dx in dx_ids:
      for proc in proc_ids:
        dp = dx + ',' + proc
        if dp not in dp_freqs:
          dp_freqs[dp] = 0
        dp_freqs[dp] += 1

    total_visit += 1

  dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.iteritems()
                  ])
  proc_probs = dict([
      (k, v / float(total_visit)) for k, v in proc_freqs.iteritems()
  ])
  dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.iteritems()
                  ])

  dp_cond_probs = {}
  pd_cond_probs = {}
  for dx, dx_prob in dx_probs.iteritems():
    for proc, proc_prob in proc_probs.iteritems():
      dp = dx + ',' + proc
      pd = proc + ',' + dx
      if dp in dp_probs:
        dp_cond_probs[dp] = dp_probs[dp] / dx_prob
        pd_cond_probs[pd] = dp_probs[dp] / proc_prob
      else:
        dp_cond_probs[dp] = 0.0
        pd_cond_probs[pd] = 0.0

  pickle.dump(dx_probs, open(output_path + '/dx_probs.empirical.p', 'wb'), -1)
  pickle.dump(proc_probs, open(output_path + '/proc_probs.empirical.p', 'wb'),
              -1)
  pickle.dump(dp_probs, open(output_path + '/dp_probs.empirical.p', 'wb'), -1)
  pickle.dump(dp_cond_probs,
              open(output_path + '/dp_cond_probs.empirical.p', 'wb'), -1)
  pickle.dump(pd_cond_probs,
              open(output_path + '/pd_cond_probs.empirical.p', 'wb'), -1)


def add_sparse_prior_guide_dp(seqex_list,
                              stats_path,
                              key_set=None,
                              max_num_codes=50):
  print('Loading conditional probabilities.')
  dp_cond_probs = pickle.load(
      open(stats_path + '/dp_cond_probs.empirical.p', 'rb'))
  pd_cond_probs = pickle.load(
      open(stats_path + '/pd_cond_probs.empirical.p', 'rb'))

  print('Adding prior guide.')
  total_visit = 0
  new_seqex_list = []
  for seqex in seqex_list:
    if total_visit % 1000 == 0:
      sys.stdout.write('Visit count: %d\r' % total_visit)
      sys.stdout.flush()

    key = seqex.context.feature['patientId'].bytes_list.value[0]
    if (key_set is not None and key not in key_set):
      total_visit += 1
      continue

    dx_ids = seqex.feature_lists.feature_list['dx_ids'].feature[
        0].bytes_list.value
    proc_ids = seqex.feature_lists.feature_list['proc_ids'].feature[
        0].bytes_list.value

    indices = []
    values = []
    for i, dx in enumerate(dx_ids):
      for j, proc in enumerate(proc_ids):
        dp = dx + ',' + proc
        indices.append((i, max_num_codes + j))
        prob = 0.0 if dp not in dp_cond_probs else dp_cond_probs[dp]
        values.append(prob)

    for i, proc in enumerate(proc_ids):
      for j, dx in enumerate(dx_ids):
        pd = proc + ',' + dx
        indices.append((max_num_codes + i, j))
        prob = 0.0 if pd not in pd_cond_probs else pd_cond_probs[pd]
        values.append(prob)

    indices = list(np.array(indices).reshape([-1]))
    indices_feature = seqex.feature_lists.feature_list['prior_indices']
    indices_feature.feature.add().int64_list.value.extend(indices)
    values_feature = seqex.feature_lists.feature_list['prior_values']
    values_feature.feature.add().float_list.value.extend(values)

    new_seqex_list.append(seqex)
    total_visit += 1

  return new_seqex_list


def main(input_path='.', output_path='.', num_fold=5):
  patient_file = input_path + '/patient.csv'
  admission_dx_file = input_path + '/admissionDx.csv'
  diagnosis_file = input_path + '/diagnosis.csv'
  treatment_file = input_path + '/treatment.csv'

  encounter_dict = {}
  print('Processing patient.csv')
  encounter_dict = process_patient(
      patient_file, encounter_dict, hour_threshold=24)
  print('Processing admission diagnosis.csv')
  encounter_dict = process_admission_dx(admission_dx_file, encounter_dict)
  print('Processing diagnosis.csv')
  encounter_dict = process_diagnosis(diagnosis_file, encounter_dict)
  print('Processing treatment.csv')
  encounter_dict = process_treatment(treatment_file, encounter_dict)

  key_list, seqex_list = build_seqex(
      encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=50)

  for i in range(num_fold):
    fold_path = output_path + '/fold_' + str(i)
    stats_path = fold_path + '/train_stats'
    os.makedirs(stats_path)

    key_train, key_valid, key_test = select_train_valid_test(
        key_list, random_seed=i)

    count_conditional_prob_dp(seqex_list, stats_path, set(key_train))
    train_seqex = add_sparse_prior_guide_dp(
        seqex_list, stats_path, set(key_train), max_num_codes=50)
    validation_seqex = add_sparse_prior_guide_dp(
        seqex_list, stats_path, set(key_valid), max_num_codes=50)
    test_seqex = add_sparse_prior_guide_dp(
        seqex_list, stats_path, set(key_test), max_num_codes=50)

    pickle.dump(train_seqex, open(fold_path + '/train.seqex', 'wb'), -1)
    pickle.dump(validation_seqex, open(fold_path + '/validation.seqex', 'wb'),
                -1)
    pickle.dump(test_seqex, open(fold_path + '/test.seqex', 'wb'), -1)


if __name__ == '__main__':
  main(input_path='.', output_path='.', num_fold=5)
