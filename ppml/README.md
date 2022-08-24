# PPML (Privacy Preserving Machine Learning)

C++ SGX/TDX attestation module for PPML

## Requirements

You can get PPML module by building locally or downloading from  
1. [Install Intel SGX SDK](https://github.com/intel/linux-sgx#install-the-intelr-sgx-sdk)

2. Install SGX DCAP verification libs

```bash
# Ubuntu 18.04, root
echo "deb [trusted=yes arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu bionic main" > etc/apt/sources.list.d/intel-sgx.list
wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add
apt update
apt -y install libsgx-dcap-quote-verify libsgx-dcap-quote-verify-dev
```

## Usage

```bash
mvn clean package
```

## (Optional) Rebuild after change JNI

Update JNI header

```bash
mvn clean package -DskipTests
javah -cp ppml-java-x86_64-linux/target/ppml-java-x86_64-linux-2.1.0-SNAPSHOT.jar com.intel.analytics.bigdl.ppml.dcap.Attestation
cp com_intel_analytics_bigdl_ppml_dcap_Attestation.h src/main/cpp
```

Check if shared lib is package into jar

```bash
mvn clean package
jar -tf ppml-java-x86_64-linux/target/ppml-java-x86_64-linux-2.1.0-SNAPSHOT.jar | grep libquote_verification.so
```

## Reference

* The `quote_verification.cpp` is tailored from [SGXDataCenterAttestationPrimitives/SampleCode/QuoteVerificationSample/App/App.cpp](https://github.com/intel/SGXDataCenterAttestationPrimitives/blob/master/SampleCode/QuoteVerificationSample/App)
