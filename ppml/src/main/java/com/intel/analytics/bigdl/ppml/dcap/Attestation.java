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

package com.intel.analytics.bigdl.ppml.dcap;

import java.util.*;

public class Attestation {
    private static boolean _isLoaded = false;

    static {
        try {
            Loader loader = new Loader();
            loader.init();

            _isLoaded = true;
        } catch (Exception e) {
            _isLoaded = false;

            e.printStackTrace();
            throw new RuntimeException("Failed to load PPML Native");
        }
    }

    /**
     * Check if shared lib is loaded
     * @return boolean
     */
    public static boolean isAttestationLoaded() {
        return _isLoaded;
    }

    public native static int sdkVerifyQuote(byte[] quote);

    public native static byte[] tdxGenerateQuote();
}
