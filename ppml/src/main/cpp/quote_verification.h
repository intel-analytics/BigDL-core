#ifndef QUOTE_VERIFICATION_H
#define QUOTE_VERIFICATION_H
#include <jni.h>

#ifdef __cplusplus
  extern "C" {
#endif


int ecdsa_quote_verification(std::vector<uint8_t> quote);

/*
 * Class:     com_intel_analytics_bigdl_ppml_attestation_Attestation
 * Method:    sdkVerifyQuote
 * Signature: ([B)I
 */
JNIEXPORT jint JNICALL Java_com_intel_analytics_bigdl_ppml_attestation_Attestation_sdkVerifyQuote
  (JNIEnv *, jclass, jbyteArray);


#ifdef __cplusplus
  }
#endif

#endif /*QUOTE_VERIFICATION_H*/

