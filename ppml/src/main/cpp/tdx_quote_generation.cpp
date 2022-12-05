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

#include "com_intel_analytics_bigdl_ppml_dcap_Attestation.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "tdx_attest.h"

#define devname         "/dev/tdx-attest"

#define HEX_DUMP_SIZE   16
#define MAX_ROW_SIZE    70

static void print_hex_dump(const char *title, const char *prefix_str,
                const uint8_t *buf, int len)
{
        const uint8_t *ptr = buf;
        int i, rowsize = HEX_DUMP_SIZE;

        if (!len || !buf)
                return;

        fprintf(stdout, "\t\t%s", title);

        for (i = 0; i < len; i++) {
                if (!(i % rowsize))
                        fprintf(stdout, "\n%s%.8x:", prefix_str, i);
                if (ptr[i] <= 0x0f)
                        fprintf(stdout, " 0%x", ptr[i]);
                else
                        fprintf(stdout, " %x", ptr[i]);
        }

        fprintf(stdout, "\n");
}

void gen_report_data(uint8_t *reportdata)
{
        int i;

        srand(time(NULL));

        for (i = 0; i < TDX_REPORT_DATA_SIZE; i++)
                reportdata[i] = rand();
}

JNIEXPORT jbyteArray JNICALL
Java_com_intel_analytics_bigdl_ppml_dcap_Attestation_tdxGenerateQuote(
    JNIEnv *env, jclass cls) {
    uint32_t quote_size = 0;
    tdx_report_data_t report_data = {{0}};
    tdx_report_t tdx_report = {{0}};
    tdx_uuid_t selected_att_key_id = {0};
    uint8_t *p_quote_buf = NULL;
    FILE *fptr = NULL;

    gen_report_data(report_data.d);
    //print_hex_dump("\n\t\tTDX report data\n", " ", report_data.d, sizeof(report_data.d));

    if (TDX_ATTEST_SUCCESS != tdx_att_get_report(&report_data, &tdx_report)) {
        fprintf(stderr, "\nFailed to get the report\n");
        return NULL;
    }
    //print_hex_dump("\n\t\tTDX report\n", " ", tdx_report.d, sizeof(tdx_report.d));

    if (TDX_ATTEST_SUCCESS != tdx_att_get_quote(&report_data, NULL, 0, &selected_att_key_id,
        &p_quote_buf, &quote_size, 0)) {
        fprintf(stderr, "\nFailed to get the quote\n");
        return NULL;
    }
    //print_hex_dump("\n\t\tTDX quote data\n", " ", p_quote_buf, quote_size);

    fprintf(stdout, "\nSuccessfully get the TD Quote\n");
    fptr = fopen("quote.dat","wb");
    if( fptr )
    {
        fwrite(p_quote_buf, quote_size, 1, fptr);
        fclose(fptr);
    }
    fprintf(stdout, "\nWrote TD Quote to quote.dat\n");

    jbyteArray ret = env->NewByteArray(quote_size);
    env->SetByteArrayRegion(ret, 0, quote_size, (jbyte*) p_quote_buf);

    tdx_att_free_quote(p_quote_buf);
    return ret;
}
