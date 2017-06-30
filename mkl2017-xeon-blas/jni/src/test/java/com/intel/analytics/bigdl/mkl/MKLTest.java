package com.intel.analytics.bigdl.mkl;

import org.junit.Test;

import static org.junit.Assert.*;


public class MKLTest {
    @Test
    public void isMKLLoaded() throws Exception {
        assertTrue(MKL.isMKLLoaded());
    }

    @Test
    public void getMklNumThreads() throws Exception {
        assertTrue(1 == MKL.getMklNumThreads());
        assertTrue(1 == MKL.getNumThreads());

        System.setProperty("bigdl.mklNumThreads", "10");
        assertTrue(10 == MKL.getMklNumThreads());
        System.clearProperty("bigdl.mklNumThreads");
    }

    @Test
    public void getMklBlockTime() throws Exception {
        assertTrue(0 == MKL.getMklBlockTime());

        System.setProperty("bigdl.mklBlockTime", "30");
        assertTrue(30 == MKL.getMklBlockTime());
        System.clearProperty("bigdl.mklBlockTime");
    }

    @Test
    public void getMklDisableFastMM() throws Exception {
        assertTrue(true == MKL.getMklDisableFastMM());

        System.setProperty("bigdl.mklDisableFastMM", "false");
        assertTrue(false == MKL.getMklDisableFastMM());
        System.clearProperty("bigdl.mklDisableFastMM");

        System.setProperty("bigdl.mklDisableFastMM", "error");
        assertTrue(true == MKL.getMklDisableFastMM());
        System.clearProperty("bigdl.mklDisableFastMM");
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

}