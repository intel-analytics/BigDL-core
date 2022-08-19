#!/bin/bash

set -ex

if [ $PPML_BUILD = true ]; then 
    make 
else
    rm -f ./ppml-target.tar.gz
    wget http://10.239.45.10:8081/repository/raw/bigdl-core/ppml-target.tar.gz
    mkdir ppml-target
    tar -zxvf ./ppml-target.tar.gz -C ppml-target
    rm -rf ./target
    mv ./ppml-target/target/ ./
    rm -rf ppml-target
    rm -f ppml-target.tar.gz
fi
