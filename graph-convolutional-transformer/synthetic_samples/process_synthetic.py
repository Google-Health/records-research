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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import sys, os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def init_dx_probs(dx_vocab_size, pareto_prior=2., pareto_prior2=1.5):
  dx_logits = np.random.pareto(pareto_prior, dx_vocab_size)
  dx_probs = dx_logits / dx_logits.sum()

  dx_cond_logits = []
  for _ in range(dx_vocab_size):
    logits = np.random.pareto(pareto_prior2, dx_vocab_size)
    dx_cond_logits.append(np.random.permutation(logits))

  dx_cond_logits = np.array(dx_cond_logits)
  for i in range(dx_vocab_size):
    dx_cond_logits[i, i] = 0.
  dx_cond_probs = dx_cond_logits / dx_cond_logits.sum(axis=1)[:, None]

  return dx_probs, dx_cond_probs


def init_dx_proc_probs(dx_vocab_size, proc_vocab_size, pareto_prior=1.5):
  dx_proc_logits = []
  for _ in range(dx_vocab_size):
    logits = np.random.pareto(pareto_prior, proc_vocab_size)
    dx_proc_logits.append(np.random.permutation(logits))

  dx_proc_logits = np.array(dx_proc_logits)
  dx_proc_probs = dx_proc_logits / dx_proc_logits.sum(axis=1)[:, None]

  return dx_proc_probs


def init_dx_proc_lab_probs(dx_vocab_size, proc_vocab_size, lab_vocab_size,
                           dx_proc_cond_probs, pareto_prior=1.5):
  dx_proc_lab_logits = []
  for i in range(dx_vocab_size):
    proc_lab_logits = []
    for j in range(proc_vocab_size):
      if dx_proc_cond_probs[i, j] == 0.0:
        proc_lab_logits.append(np.zeros(lab_vocab_size))
      else:
        logits = np.random.pareto(pareto_prior, lab_vocab_size)
        proc_lab_logits.append(np.random.permutation(logits))
    dx_proc_lab_logits.append(proc_lab_logits)
  dx_proc_lab_logits = np.array(dx_proc_lab_logits)
  dx_proc_lab_probs = (
      dx_proc_lab_logits / (dx_proc_lab_logits.sum(axis=2)[:, :, None] + 1e-12))

  return dx_proc_lab_probs


def generate_dx(dx_probs, dx_dx_cond_probs, multi_dx_prob, dx_dx_probs):
  dx_list = []
  while np.random.uniform(0., 1.) < multi_dx_prob or len(dx_list) < 2:
    new_dx = np.argmax(np.random.multinomial(1, dx_probs))
    dx_list.append(new_dx)
    prev_dx = new_dx
    while np.random.uniform(0., 1.) < dx_dx_probs[prev_dx]:
      new_dx = np.argmax(np.random.multinomial(1, dx_dx_cond_probs[prev_dx]))
      dx_list.append(new_dx)
      prev_dx = new_dx
  return list(set(dx_list))


def generate_procs(dx, dx_proc_cond_probs, multi_proc_probs,
                   dx_proc_lab_cond_probs, multi_lab_probs):
  proc_list = []
  proc_set = set()
  while np.random.uniform(0., 1.) < multi_proc_probs[dx]:
    proc = np.argmax(np.random.multinomial(1, dx_proc_cond_probs[dx]))
    if proc in proc_set:
      continue
    proc_set.add(proc)
    labs = generate_labs(dx, proc, dx_proc_lab_cond_probs, multi_lab_probs)
    proc_list.append([proc, labs])
  return proc_list


def generate_labs(dx, proc, dx_proc_lab_cond_probs, multi_lab_probs):
  lab_list = []
  lab_set = set()
  while np.random.uniform(0., 1.) < multi_lab_probs[dx, proc]:
    lab = np.argmax(np.random.multinomial(1, dx_proc_lab_cond_probs[dx, proc]))
    if lab in lab_set:
      continue
    lab_set.add(lab)
    lab_list.append(lab)
  return lab_list


def generate_visit(
    dx_probs,
    dx_dx_cond_probs,
    dx_proc_cond_probs,
    dx_proc_lab_cond_probs,
    multi_dx_prob,
    dx_dx_probs,
    multi_proc_probs,
    multi_lab_probs):
  visit = []
  dx_list = generate_dx(dx_probs, dx_dx_cond_probs, multi_dx_prob, dx_dx_probs)
  for dx in dx_list:
    proc_labs = generate_procs(dx, dx_proc_cond_probs, multi_proc_probs,
                               dx_proc_lab_cond_probs, multi_lab_probs)
    visit.append([dx, proc_labs])

  return visit


def build_seqex(
    visits,
    dp_chains=['dx_199,proc_939', 'dx_133,proc_939'],
    easy_labels=['dx_199', 'dx_134'],
    min_num_codes=5,
    max_num_codes=50):
  key_list = []
  seqex_list = []
  dx_str2int = {}
  proc_str2int = {}
  lab_str2int = {}
  easy_label_count = 0

  for i, visit in enumerate(visits):
    seqex = tf.train.SequenceExample()
    patient_id = str(i)
    seqex.context.feature['patientId'].bytes_list.value.append(patient_id)
    seqex.context.feature['sequenceLength'].int64_list.value.append(1)
    key = ('Patient/%s:0-1@0:Encounter/0' % patient_id)

    vd_pair = []
    dp_pair = []
    pl_pair = []

    dx_list = []
    proc_list = []
    lab_list = []
    dp_chain_label_list = []

    dx_map = {}
    proc_map = {}
    lab_map = {}

    for dx_obj in visit:
      dx = 'dx_' + str(dx_obj[0])
      if dx not in dx_str2int: dx_str2int[dx] = len(dx_str2int)
      dx_map[dx] = len(dx_map)
      dx_list.append(dx)
      vd_pair.append((0, dx_map[dx]))


      for proc_obj in dx_obj[1]:
        proc = 'proc_' + str(proc_obj[0])
        if proc not in proc_str2int: proc_str2int[proc] = len(proc_str2int)
        if proc not in proc_map:
          proc_map[proc] = len(proc_map)
          proc_list.append(proc)
        dp_pair.append((dx_map[dx], proc_map[proc]))

        dp_chain = dx + ',' + proc
        if dp_chain in dp_chains:
          dp_chain_label_list.append(dp_chain)

        for lab_obj in proc_obj[1]:
          lab = 'loinc_' + str(lab_obj)
          if lab not in lab_str2int: lab_str2int[lab] = len(lab_str2int)
          if lab not in lab_map:
            lab_map[lab] = len(lab_map)
            lab_list.append(lab)
          pl_pair.append((proc_map[proc], lab_map[lab]))

    if (len(dx_list) < min_num_codes or len(proc_list) < min_num_codes or
        len(lab_list) < min_num_codes):
      continue

    if (len(dx_list) > max_num_codes or len(proc_list) > max_num_codes or
        len(lab_list) > max_num_codes):
      continue

    dx_ids = seqex.feature_lists.feature_list['dx_ids']
    dx_ids.feature.add().bytes_list.value.extend(dx_list)

    dx_int_list = [dx_str2int[item] for item in dx_list]
    dx_ints = seqex.feature_lists.feature_list['dx_ints']
    dx_ints.feature.add().int64_list.value.extend(dx_int_list)

    proc_ids = seqex.feature_lists.feature_list['proc_ids']
    proc_ids.feature.add().bytes_list.value.extend(proc_list)

    proc_int_list = [proc_str2int[item] for item in proc_list]
    proc_ints = seqex.feature_lists.feature_list['proc_ints']
    proc_ints.feature.add().int64_list.value.extend(proc_int_list)

    lab_ids = seqex.feature_lists.feature_list['loinc_bucketized_values']
    lab_ids.feature.add().bytes_list.value.extend(lab_list)

    lab_int_list = [lab_str2int[item] for item in lab_list]
    lab_ints = seqex.feature_lists.feature_list['loinc_bucketized_ints']
    lab_ints.feature.add().int64_list.value.extend(lab_int_list)

    if len(dp_chain_label_list) > 0:
      seqex.context.feature['label.medication.class'].bytes_list.value.extend(list(set(dp_chain_label_list)))

    if easy_labels[0] in dx_map and easy_labels[1] in dx_map:
      seqex.context.feature['label.expired.class'].bytes_list.value.append('expired')
      easy_label_count += 1

    vd_pair = list(set(vd_pair))
    vd_pair = np.array(vd_pair).reshape((-1))
    vd_pair_feature = seqex.feature_lists.feature_list['vd_pair']
    vd_pair_feature.feature.add().int64_list.value.extend(vd_pair)

    dp_pair = list(set(dp_pair))
    dp_pair = np.array(dp_pair).reshape((-1))
    dp_pair_feature = seqex.feature_lists.feature_list['dp_pair']
    dp_pair_feature.feature.add().int64_list.value.extend(dp_pair)

    pl_pair = list(set(pl_pair))
    pl_pair = np.array(pl_pair).reshape((-1))
    pl_pair_feature = seqex.feature_lists.feature_list['pl_pair']
    pl_pair_feature.feature.add().int64_list.value.extend(pl_pair)

    key_list.append(key)
    seqex_list.append(seqex)

  print('Number of visits with easy labels: %d' % easy_label_count)
  return key_list, seqex_list, dx_str2int, proc_str2int, lab_str2int


def count_conditional_prob_dpl(key_list, seqex_list, output_path, train_key_set=None):
  dx_freqs = {}
  proc_freqs = {}
  lab_freqs = {}
  dp_freqs = {}
  pl_freqs = {}
  total_visit = 0

  for key, seqex in zip(key_list, seqex_list):
    if total_visit % 1000 == 0:
      sys.stdout.write('Visit count: %d\r' % total_visit)
      sys.stdout.flush()

    if train_key_set is not None and key not in train_key_set:
      total_visit += 1
      continue

    dx_ids = seqex.feature_lists.feature_list['dx_ids'].feature[0].bytes_list.value
    proc_ids = seqex.feature_lists.feature_list['proc_ids'].feature[0].bytes_list.value
    lab_ids = seqex.feature_lists.feature_list['loinc_bucketized_values'].feature[0].bytes_list.value

    for dx in dx_ids:
      if dx not in dx_freqs:
        dx_freqs[dx] = 0
      dx_freqs[dx] += 1

    for proc in proc_ids:
      if proc not in proc_freqs:
        proc_freqs[proc] = 0
      proc_freqs[proc] += 1

    for lab in lab_ids:
      if lab not in lab_freqs:
        lab_freqs[lab] = 0
      lab_freqs[lab] += 1

    for dx in dx_ids:
      for proc in proc_ids:
        dp = dx + ',' + proc
        if dp not in dp_freqs:
          dp_freqs[dp] = 0
        dp_freqs[dp] += 1

    for proc in proc_ids:
      for lab in lab_ids:
        pl = proc + ',' + lab
        if pl not in pl_freqs:
          pl_freqs[pl] = 0
        pl_freqs[pl] += 1

    total_visit += 1

  dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.iteritems()])
  proc_probs = dict([(k, v / float(total_visit)) for k, v in proc_freqs.iteritems()])
  lab_probs = dict([(k, v / float(total_visit)) for k, v in lab_freqs.iteritems()])
  dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.iteritems()])
  pl_probs = dict([(k, v / float(total_visit)) for k, v in pl_freqs.iteritems()])

  dp_cond_probs = {}
  pd_cond_probs = {}
  for dx in dx_probs.keys():
    for proc in proc_probs.keys():
      dp = dx + ',' + proc
      pd = proc + ',' + dx
      if dp in dp_probs:
        dp_cond_probs[dp] = dp_probs[dp] / dx_probs[dx]
        pd_cond_probs[pd] = dp_probs[dp] / proc_probs[proc]
      else:
        dp_cond_probs[dp] = 0.0
        pd_cond_probs[pd] = 0.0

  pl_cond_probs = {}
  lp_cond_probs = {}
  for proc in proc_probs.keys():
    for lab in lab_probs.keys():
      pl = proc + ',' + lab
      lp = lab + ',' + proc
      if pl in pl_probs:
        pl_cond_probs[pl] = pl_probs[pl] / proc_probs[proc]
        lp_cond_probs[lp] = pl_probs[pl] / lab_probs[lab]
      else:
        pl_cond_probs[pl] = 0.0
        pl_cond_probs[lp] = 0.0

  pickle.dump(dx_probs, open(output_path + '/dx_probs.empirical.p', 'wb'), -1)
  pickle.dump(proc_probs, open(output_path + '/proc_probs.empirical.p', 'wb'), -1)
  pickle.dump(lab_probs, open(output_path + '/lab_probs.empirical.p', 'wb'), -1)
  pickle.dump(dp_probs, open(output_path + '/dp_probs.empirical.p', 'wb'), -1)
  pickle.dump(pl_probs, open(output_path + '/pl_probs.empirical.p', 'wb'), -1)
  pickle.dump(dp_cond_probs, open(output_path + '/dp_cond_probs.empirical.p', 'wb'), -1)
  pickle.dump(pd_cond_probs, open(output_path + '/pd_cond_probs.empirical.p', 'wb'), -1)
  pickle.dump(pl_cond_probs, open(output_path + '/pl_cond_probs.empirical.p', 'wb'), -1)
  pickle.dump(lp_cond_probs, open(output_path + '/lp_cond_probs.empirical.p', 'wb'), -1)


def add_sparse_prior_guide_dpl(key_list, seqex_list, stats_path, key_set=None, max_num_codes=50):
  print('Loading conditional probabilities.')
  dp_cond_probs = pickle.load(open(stats_path + '/dp_cond_probs.empirical.p', 'rb'))
  pd_cond_probs = pickle.load(open(stats_path + '/pd_cond_probs.empirical.p', 'rb'))
  pl_cond_probs = pickle.load(open(stats_path + '/pl_cond_probs.empirical.p', 'rb'))
  lp_cond_probs = pickle.load(open(stats_path + '/lp_cond_probs.empirical.p', 'rb'))

  print('Adding prior guide.')
  total_visit = 0
  new_seqex_list = []
  for key, seqex in zip(key_list, seqex_list):
    if total_visit % 1000 == 0:
      sys.stdout.write('Visit count: %d\r' % total_visit)
      sys.stdout.flush()

    if key_set is not None and key not in key_set:
      total_visit += 1
      continue

    dx_ids = seqex.feature_lists.feature_list['dx_ids'].feature[0].bytes_list.value
    proc_ids = seqex.feature_lists.feature_list['proc_ids'].feature[0].bytes_list.value
    lab_ids = seqex.feature_lists.feature_list['loinc_bucketized_values'].feature[0].bytes_list.value

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

    for i, proc in enumerate(proc_ids):
      for j, lab in enumerate(lab_ids):
        pl = proc + ',' + lab
        indices.append((max_num_codes + i, max_num_codes * 2 + j))
        prob = 0.0 if pl not in pl_cond_probs else pl_cond_probs[pl]
        values.append(prob)

    for i, lab in enumerate(lab_ids):
      for j, proc in enumerate(proc_ids):
        lp = lab + ',' + proc
        indices.append((max_num_codes * 2 + i, max_num_codes + j))
        prob = 0.0 if lp not in lp_cond_probs else lp_cond_probs[lp]
        values.append(prob)

    indices = list(np.array(indices).reshape([-1]))
    indices_feature = seqex.feature_lists.feature_list['prior_indices']
    indices_feature.feature.add().int64_list.value.extend(indices)
    values_feature = seqex.feature_lists.feature_list['prior_values']
    values_feature.feature.add().float_list.value.extend(values)

    new_seqex_list.append(seqex)
    total_visit += 1

  return new_seqex_list


def select_train_valid_test(key_list, random_seed=1234):
  key_train, key_temp = train_test_split(key_list, test_size=0.2, random_state=random_seed)
  key_valid, key_test = train_test_split(key_temp, test_size=0.5, random_state=random_seed)
  return key_train, key_valid, key_test


def sample_simulated(argv):
  output_path = argv[1]
  num_visits = 2000000
  dx_vocab_size = 1000
  proc_vocab_size = 1000
  lab_vocab_size = 1000

  print('Intializing code occurrence probabilities.')
  # How likely independent Dx codes occur?
  multi_dx_prob = 0.5

  # How likely is one Dx to lead to another Dx?
  dx_dx_probs = np.random.normal(0.5, 0.1, size=dx_vocab_size)
  dx_dx_probs[dx_dx_probs < 0.] = 0.
  dx_dx_probs[dx_dx_probs > 0.99] = 0.5

  # How many procedures are likely to occur based on Dx?
  multi_proc_probs = np.random.normal(0.5, 0.25, size=dx_vocab_size)
  multi_proc_probs[multi_proc_probs < 0.] = 0.
  multi_proc_probs[multi_proc_probs > 0.99] = 0.5

  # How many labs are likely to occur based on Dx and Proc?
  multi_lab_probs = np.random.normal(
      0.5, 0.25, size=[dx_vocab_size, proc_vocab_size])
  multi_lab_probs[multi_lab_probs < 0.] = 0.
  multi_lab_probs[multi_lab_probs > 0.99] = 0.5

  print('Intializing conditional code probabilities.')
  dx_probs, dx_dx_cond_probs = init_dx_probs(
      dx_vocab_size, pareto_prior=2., pareto_prior2=1.5)
  dx_proc_cond_probs = init_dx_proc_probs(
      dx_vocab_size, proc_vocab_size, pareto_prior=1.5)
  dx_proc_lab_cond_probs = init_dx_proc_lab_probs(
      dx_vocab_size, proc_vocab_size, lab_vocab_size, dx_proc_cond_probs,
      pareto_prior=1.5)

  np.save(output_path + '/dx_dx_probs', dx_dx_probs)
  np.save(output_path + '/multi_proc_probs', multi_proc_probs)
  np.save(output_path + '/multi_lab_probs', multi_lab_probs)
  np.save(output_path + '/dx_probs', dx_probs)
  np.save(output_path + '/dx_dx_cond_probs', dx_dx_cond_probs)
  np.save(output_path + '/dx_proc_cond_probs', dx_proc_cond_probs)
  np.save(output_path + '/dx_proc_lab_cond_probs', dx_proc_lab_cond_probs)

  print('Generating visits.')
  visit_list = []
  for i in range(num_visits):
    if i % 100 == 0:
      sys.stdout.write('%d\r' % i)
      sys.stdout.flush()
    visit = generate_visit(
        dx_probs,
        dx_dx_cond_probs,
        dx_proc_cond_probs,
        dx_proc_lab_cond_probs,
        multi_dx_prob,
        dx_dx_probs,
        multi_proc_probs,
        multi_lab_probs)
    visit_list.append(visit)
    i += 1

  return visit_list


"""
Set <input_path> to where the visit_list.p file is located.
Set <output_path> to where you want the TFRecords to be stored.
"""
def seqex_main(argv):
  input_path = argv[1]
  output_path = argv[2]
  num_fold = 5
  data_size = 50000

  print('Reading visit_list.')
  visit_list = pickle.load(open(input_path + '/visit_list.p', 'rb'))

  print('Converting to SequenceExamples.')
  key_list, seqex_list, dx_map, proc_map, lab_map = build_seqex(visit_list)
  pickle.dump(dx_map, open(output_path + '/dx_map.p', 'wb'), -1)
  pickle.dump(proc_map, open(output_path + '/proc_map.p', 'wb'), -1)
  pickle.dump(lab_map, open(output_path + '/lab_map.p', 'wb'), -1)

  for i in range(num_fold):
    print('Creating fold %d/%d.' % (i, num_fold))
    fold_path = output_path + '/fold_' + str(i)
    stats_path = fold_path + '/train_stats'
    os.makedirs(stats_path)

    key_train, key_valid, key_test = select_train_valid_test(key_list[:data_size], random_seed=i)
    pickle.dump(key_train, open(fold_path + '/train_key_list.p', 'wb'), -1)
    pickle.dump(key_valid, open(fold_path + '/validation_key_list.p', 'wb'), -1)
    pickle.dump(key_test, open(fold_path + '/test_key_list.p', 'wb'), -1)

    count_conditional_prob_dpl(key_list, seqex_list, stats_path, set(key_train))
    train_seqex = add_sparse_prior_guide_dpl(key_list, seqex_list, stats_path, set(key_train), max_num_codes=50)
    validation_seqex = add_sparse_prior_guide_dpl(key_list, seqex_list, stats_path, set(key_valid), max_num_codes=50)
    test_seqex = add_sparse_prior_guide_dpl(key_list, seqex_list, stats_path, set(key_test), max_num_codes=50)

    with tf.io.TFRecordWriter(fold_path + '/train.tfrecord') as writer:
      for seqex in train_seqex:
        writer.write(seqex.SerializeToString())

    with tf.io.TFRecordWriter(fold_path + '/validation.tfrecord') as writer:
      for seqex in validation_seqex:
        writer.write(seqex.SerializeToString())

    with tf.io.TFRecordWriter(fold_path + '/test.tfrecord') as writer:
      for seqex in test_seqex:
        writer.write(seqex.SerializeToString())


if __name__ == '__main__':
    seqex_main(sys.argv)
