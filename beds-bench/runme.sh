#!/bin/bash
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


set -e

PHYSIONET_USER=

JSON=config.json
WORK_DIR=`grep work-dir config.json | cut -f2 -d: | cut -f2 -d'"'`

MIMIC_DIR=$WORK_DIR/mimic-iii-clinical-database-1.4
MIMIC_ZIP=$MIMIC_DIR.zip
MIMIC_URL=https://physionet.org/files/mimiciii/1.4/

PICDB_DIR=$WORK_DIR/paediatric-intensive-care-database-1.0.0
PICDB_ZIP=$PICDB_DIR.zip
PICDB_URL=https://physionet.org/files/picdb/1.0.0/

function set_physionet_user(){
  if [ "x$PHYSIONET_USER" = "x" ]; then
    echo -n "Enter Physionet username: "
    read PHYSIONET_USER
  fi
  if [ "x$PHYSIONET_USER" = "x" ]; then
    echo "!!!"
    echo "!!! Set the PHYSIONET_USER= variable and re-run"
    echo "!!!"
    exit 1
  fi
}

function download_mimic(){
  if [ -f $MIMIC_ZIP ]; then
    echo "MIMIC-III v1.4 zip found, skipping download."
    return
  fi
  echo "MIMIC-III not found, downloading..."
  set_physionet_user
  wget -N -c -np --user "$PHYSIONET_USER" --ask-password $MIMIC_URL -o $PICDB_ZIP
}

function download_picdb(){
  if [ -f $PICDB_ZIP ]; then
    echo "PICDB v1.0.0 zip found, skipping download."
    return
  fi
  echo "PICDB not found, downloading..."
  set_physionet_user
  wget -N -c -np --user "$PHYSIONET_USER" --ask-password $PICDB_URL -o $PICDB_ZIP
}

function extract_mimic(){
  if [ -f $MIMIC_DIR/TRANSFERS.csv.gz -o -f $MIMIC_DIR/TRANSFERS.csv ]; then
    echo "MIMIC already extracted, skipping."
    return
  fi
  download_mimic
  unzip -d $WORK_DIR $MIMIC_ZIP
}

function extract_picdb(){
  if [ -f $PICDB_DIR/SURGERY_VITAL_SIGNS.csv.gz -o -f $PICDB_DIR/SURGERY_VITAL_SIGNS.csv ]; then
    echo "PICDB already extracted, skipping."
    return 0
  fi
  download_picdb
  unzip -d $WORK_DIR $PICDB_ZIP
}

function main(){
  mkdir -p $WORK_DIR
  extract_mimic
  extract_picdb

  python prepare.py
  python feat_fixedlen.py
  python benchmark.py
  python results.py
}

main "$@"
