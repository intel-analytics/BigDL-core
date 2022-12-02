# PPML-TDX (Privacy Preserving Machine Learning)

C++ TDX attestation module for PPML

## Requirements
 
1. [Install Intel SGX SDK](https://github.com/intel/linux-sgx#install-the-intelr-sgx-sdk)

2. Install `libtdx-attest`

```bash
# Centos, root
yum install -y yum-utils && \
yum-config-manager --add-repo \
https://enclave-cn-shanghai.oss-cn-shanghai.aliyuncs.com/repo/alinux/enclave-expr.repo

yum install libtdx-attest libtdx-attest-devel
```

## Usage

Configure environment variables `SGX_SDK`, `JAVA_HOME`, `PPML_BUILD` correctly. Then build jar with command:
```bash
export PPML_BUILD=true

mvn clean package
```
