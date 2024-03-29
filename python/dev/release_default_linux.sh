#!/usr/bin/env bash

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This is the default script with maven parameters to release bigdl-tf for linux.
# Note that if the maven parameters to build bigdl-tf need to be changed,
# make sure to change this file accordingly.
# If you want to customize the release, please use release.sh and specify maven parameters instead.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR

if (( $# < 3)); then
  echo "Usage: release_default_linux.sh bigdl-core-jar-path version upload"
  echo "Usage example: bash release_default_linux.sh all-xxx.jar default true"
  echo "Usage example: bash release_default_linux.sh all-xxx.jar 0.14.0.dev1 true"
  echo "Usage example: bash release_default_linux.sh http://all-xxx.jar 0.14.0.dev1 true"
  exit -1
fi

path=$1
version=$2
upload=$3

bash ${RUN_SCRIPT_DIR}/release.sh linux ${path} ${version} ${upload}
