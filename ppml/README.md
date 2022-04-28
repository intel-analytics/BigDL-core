# PPML (Privacy Preserving Machine Learning)

C++ module for PPML

## Usage

```bash
mvn clean compile
```

Generate Attestation C++ header

```bash
javah -cp ppml-java-x86_64-linux/target/ppml-java-x86_64-linux-2.1.0-SNAPSHOT.jar com.intel.analytics.bigdl.ppml.attestation.Attestation
```

## Reference
* The `quote_verification.cpp` is tailored from [SGXDataCenterAttestationPrimitives/SampleCode/QuoteVerificationSample/App/App.cpp](https://github.com/intel/SGXDataCenterAttestationPrimitives/blob/master/SampleCode/QuoteVerificationSample/App)

