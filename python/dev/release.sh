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

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
export BIGDL_CORE_DIR="$(cd ${RUN_SCRIPT_DIR}/../..; pwd)"
echo $BIGDL_CORE_DIR
BIGDL_PYTHON_DIR="$(cd ${BIGDL_CORE_DIR}/python/src; pwd)"
echo $BIGDL_PYTHON_DIR
BIGDL_PYTHON_LIB_DIR="$(cd ${BIGDL_CORE_DIR}/python/src/bigdl/share/core/lib; pwd)"
echo $BIGDL_PYTHON_LIB_DIR


if (( $# < 4)); then
  echo "Usage: release.sh platform jar-path version upload"
  echo "Usage example: bash release.sh linux all-xxx.jar default true"
  echo "Usage example: bash release.sh mac all-xxx.jar 0.14.0.dev1 false"
  exit -1
fi

platform=$1
jarpath=$2
version=$3
upload=$4  # Whether to upload the whl to pypi

if [ "${version}" != "default" ]; then
    echo "User specified version: ${version}"
    echo $version > $BIGDL_CORE_DIR/python/version.txt
fi
bigdl_version=$(cat $BIGDL_CORE_DIR/python/version.txt | head -1)
echo "The effective version is: ${bigdl_version}"

regex='(https?|ftp)://[-A-Za-z0-9\+&@#/%?=~_|!:,.;]*[-A-Za-z0-9\+&@#/%=~_|]'
if [[ $jarpath =~ $regex ]]
then
    wget -P $BIGDL_PYTHON_LIB_DIR $jarpath
else
    cp -f $jarpath $BIGDL_PYTHON_LIB_DIR
fi
export BIGDL_CORE_JAR=`ls $BIGDL_PYTHON_LIB_DIR/ | grep *.jar`
export BIGDL_CORE_JAR_PATH=`ls $BIGDL_PYTHON_LIB_DIR/*.jar`

if [ "$platform" ==  "mac" ]; then
    zip -d $BIGDL_CORE_JAR_PATH *.so
    verbose_pname="macosx_10_11_x86_64"
elif [ "$platform" == "linux" ]; then
    zip -d $BIGDL_CORE_JAR_PATH *.dylib
    verbose_pname="manylinux2010_x86_64"
else
    echo "Unsupported platform"
fi
#wget xin-dev.sh.intel.com/share/all-2.1.0-20220314.094552-2.jar
#mv all-2.1.0-20220314.094552-2.jar $BIGDL_CORE_JAR

cd $BIGDL_PYTHON_DIR
wheel_command="python setup.py bdist_wheel --plat-name ${verbose_pname} --python-tag py3"
echo "Packing python distribution: $wheel_command"
${wheel_command}

if [ ${upload} == true ]; then
    upload_command="twine upload dist/bigdl_core-${bigdl_version}-py3-none-${verbose_pname}.whl"
    echo "Please manually upload with this command: $upload_command"
    $upload_command
fi

# clean up
rm $BIGDL_PYTHON_LIB_DIR/*.jar*
