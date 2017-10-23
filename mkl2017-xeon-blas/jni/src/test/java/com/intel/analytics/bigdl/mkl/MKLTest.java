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

package com.intel.analytics.bigdl.mkl;

import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.*;


public class MKLTest {
    @After
    public void clearProperties() {
        System.clearProperty("bigdl.mklNumThreads");
        System.clearProperty("bigdl.mklBlockTime");
        System.clearProperty("bigdl.mklDisableFastMM");
        System.clearProperty("bigdl.mklWaitPolicy");
    }

    @Test
    public void isMKLLoaded() throws Exception {
        assertTrue(MKL.isMKLLoaded());
    }

    @Test(expected = UnsupportedOperationException.class)
    public void getMklNumThreads() throws Exception {
        assertTrue(1 == MKL.getMklNumThreads());
        assertTrue(1 == MKL.getNumThreads());

        System.setProperty("bigdl.mklNumThreads", "10");
        assertTrue(10 == MKL.getMklNumThreads());
        System.clearProperty("bigdl.mklNumThreads");

        System.setProperty("bigdl.mklNumThreads", "-1");
        MKL.getMklNumThreads();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void getMklBlockTime() throws Exception {
        assertTrue(0 == MKL.getMklBlockTime());

        System.setProperty("bigdl.mklBlockTime", "30");
        assertTrue(30 == MKL.getMklBlockTime());
        System.clearProperty("bigdl.mklBlockTime");

        System.setProperty("bigdl.mklBlockTime", "-1");
        MKL.getMklBlockTime();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void getMklDisableFastMM() throws Exception {
        assertTrue(true == MKL.getMklDisableFastMM());

        System.setProperty("bigdl.mklDisableFastMM", "false");
        assertTrue(false == MKL.getMklDisableFastMM());
        System.clearProperty("bigdl.mklDisableFastMM");

        System.setProperty("bigdl.mklDisableFastMM", "error");
        MKL.getMklDisableFastMM();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void getMklWaitPolicy() throws Exception {
        assertTrue(3 == MKL.getMklWaitPolicy());

        System.setProperty("bigdl.mklWaitPolicy", "active");
        assertTrue(2 == MKL.getMklWaitPolicy());
        System.clearProperty("bigdl.mklWaitPolicy");

        System.setProperty("bigdl.mklWaitPolicy", "error");
        MKL.getMklWaitPolicy();
    }

    @Test
    public void sgemmWithPackAMatrix() {
        int m = 4;
        int k = 5;
        int n = 6;

        float[] a = new float[m * k];
        float[] b = new float[k * n];
        float[] c1 = new float[m * n];
        float[] c2 = new float[m * n];

        for (int i = 0; i < m*k; i++) {
            a[i] = i + 1;
        }

        for (int i = 0; i < k * n; i++) {
            b[i] = i + 1;
        }
        for (int i = 0; i < m * n; i++) {
            c1[i] = 0;
            c2[i] = 0;
        }

        float alpha = 1f;
        float beta = 0f;
        int lda = m;
        int ldb = k;
        int ldc = m;

        long packMem = MKL.sgemmAlloc('A', m, n, k);
        MKL.sgemmPack('A', 'N', m, n, k, alpha, a, 0, lda, packMem);
        MKL.sgemmCompute('P', 'N', m, n, k, a, 0, lda,
                b, 0, ldb, beta, c1, 0, ldc, packMem);
        for (int i = 0; i < m * n; i++) {
            System.out.println(c1[i]);
        }
        System.out.println("*");
        MKL.sgemmFree(packMem);

        MKL.vsgemm('N', 'N', m, n, k, alpha, a, 0,
                lda, b, 0, ldb, beta, c2, 0, ldc);

        for (int i = 0; i < m * n; i++) {
            System.out.println(c2[i]);
        }

        for (int i = 0; i < m * n; i++) {
            assertTrue(c1[i] == c2[i]);
        }
    }

    @Test
    public void sgemmWithPackBMatrix() {
        int m = 4;
        int k = 5;
        int n = 6;

        float[] a = new float[m * k];
        float[] b = new float[k * n];
        float[] c1 = new float[m * n];
        float[] c2 = new float[m * n];

        for (int i = 0; i < m*k; i++) {
            a[i] = i + 1;
        }

        for (int i = 0; i < k * n; i++) {
            b[i] = i + 1;
        }
        for (int i = 0; i < m * n; i++) {
            c1[i] = 0;
            c2[i] = 0;
        }

        float alpha = 1f;
        float beta = 0f;
        int lda = m;
        int ldb = k;
        int ldc = m;

        long packMem = MKL.sgemmAlloc('B', m, n, k);
        MKL.sgemmPack('B', 'N', m, n, k, alpha, b, 0, ldb, packMem);
        MKL.sgemmCompute('N', 'P', m, n, k, a, 0, lda,
                b, 0, ldb, beta, c1, 0, ldc, packMem);
        for (int i = 0; i < m * n; i++) {
            System.out.println(c1[i]);
        }
        System.out.println("*");
        MKL.sgemmFree(packMem);

        MKL.vsgemm('N', 'N', m, n, k, alpha, a, 0,
                lda, b, 0, ldb, beta, c2, 0, ldc);

        for (int i = 0; i < m * n; i++) {
            System.out.println(c2[i]);
        }

        for (int i = 0; i < m * n; i++) {
            assertTrue(c1[i] == c2[i]);
        }
    }
}