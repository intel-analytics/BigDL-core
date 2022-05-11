/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "com_intel_analytics_bigdl_ppml_attestation_Attestation.h"
#include "sgx_dcap_quoteverify.h"
#include "sgx_ql_quote.h"
#include "sgx_urts.h"
#include <assert.h>
#include <fstream>
#include <sgx_uae_launch.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

using namespace std;

/**
 * @param quote - ECDSA quote buffer
 * @return verification result (int) 0 success, 1 failed, -1 error
 */
int ecdsa_quote_verification(vector<uint8_t> quote) {
  int ret = 0;
  time_t current_time = 0;
  uint32_t supplemental_data_size = 0;
  uint8_t *p_supplemental_data = NULL;
  quote3_error_t dcap_ret = SGX_QL_ERROR_UNEXPECTED;
  sgx_ql_qv_result_t quote_verification_result = SGX_QL_QV_RESULT_UNSPECIFIED;
  uint32_t collateral_expiration_status = 1;

  // quote verification
  // call DCAP quote verify library to get supplemental data size
  dcap_ret = sgx_qv_get_quote_supplemental_data_size(&supplemental_data_size);
  if (dcap_ret == SGX_QL_SUCCESS &&
      supplemental_data_size == sizeof(sgx_ql_qv_supplemental_t)) {
    printf("\tInfo: sgx_qv_get_quote_supplemental_data_size successfully "
           "returned.\n");
    p_supplemental_data = (uint8_t *)malloc(supplemental_data_size);
  } else {
    if (dcap_ret != SGX_QL_SUCCESS)
      printf(
          "\tError: sgx_qv_get_quote_supplemental_data_size failed: 0x%04x\n",
          dcap_ret);

    if (supplemental_data_size != sizeof(sgx_ql_qv_supplemental_t))
      printf("\tWarning: sgx_qv_get_quote_supplemental_data_size returned size "
             "is not same with header definition in SGX SDK, please make sure "
             "you are using same version of SGX SDK and DCAP QVL.\n");

    supplemental_data_size = 0;
  }

  // set current time.
  current_time = time(NULL);

  // call DCAP quote verify library for quote verification
  // here you can choose 'trusted' or 'untrusted' quote verification by
  // specifying parameter '&qve_report_info' if '&qve_report_info' is NOT NULL,
  dcap_ret = sgx_qv_verify_quote(quote.data(), (uint32_t)quote.size(), NULL,
                                 current_time, &collateral_expiration_status,
                                 &quote_verification_result, NULL,
                                 supplemental_data_size, p_supplemental_data);
  if (dcap_ret == SGX_QL_SUCCESS) {
    printf("\tInfo: sgx_qv_verify_quote successfully returned.\n");
  } else {
    printf("\tError: sgx_qv_verify_quote failed: 0x%04x\n", dcap_ret);
  }

  // check verification result
  switch (quote_verification_result) {
  case SGX_QL_QV_RESULT_OK:
    printf("\tInfo: Verification completed successfully.\n");
    ret = 0;
    break;
  case SGX_QL_QV_RESULT_CONFIG_NEEDED:
  case SGX_QL_QV_RESULT_OUT_OF_DATE:
  case SGX_QL_QV_RESULT_OUT_OF_DATE_CONFIG_NEEDED:
  case SGX_QL_QV_RESULT_SW_HARDENING_NEEDED:
  case SGX_QL_QV_RESULT_CONFIG_AND_SW_HARDENING_NEEDED:
    printf(
        "\tWarning: Verification completed with Non-terminal result: %x\n",
        quote_verification_result);
    ret = 1;
    break;
  case SGX_QL_QV_RESULT_INVALID_SIGNATURE:
  case SGX_QL_QV_RESULT_REVOKED:
  case SGX_QL_QV_RESULT_UNSPECIFIED:
  default:
    printf("\tError: Verification completed with Terminal result: %x\n",
           quote_verification_result);
    ret = -1;
    break;
  }
  return ret;
}

JNIEXPORT jint JNICALL
Java_com_intel_analytics_bigdl_ppml_attestation_Attestation_sdkVerifyQuote(
    JNIEnv *env, jclass cls, jbyteArray quote) {
  // Return -1 if quote is null
  if (quote == NULL)
    return -1;
  // convert jbyteArray to vector<char>
  jbyte *jbae = env->GetByteArrayElements(quote, 0);
  jsize len = env->GetArrayLength(quote);
  // Copy quote to quote_vector
  char *quote_arrary = (char *)jbae;
  vector<unsigned char> quote_vector;
  for (int i = 0; i < len; i++) {
    quote_vector.push_back(quote_arrary[i]);
  }
  int result = ecdsa_quote_verification(quote_vector);
  return result;
}
