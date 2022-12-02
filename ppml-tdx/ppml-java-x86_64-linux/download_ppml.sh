#!/bin/bash

set -ex

if [ $PPML_BUILD = true ]; then 
    make 
else
    rm -f ./ppml-tdx-target.tar.gz
    wget http://10.239.45.10:8081/repository/raw/bigdl-core/ppml-tdx-target.tar.gz
    mkdir ppml-tdx-target
    tar -zxvf ./ppml-tdx-target.tar.gz -C ppml-tdx-target
    rm -rf ./target
    mv ./ppml-tdx-target/target/ ./
    rm -rf ppml-tdx-target
    rm -f ppml-tdx-target.tar.gz
fi
