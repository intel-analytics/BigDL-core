package com.intel.analytics.bigdl.mkl.hardware;

import com.intel.analytics.bigdl.mkl.hardware.platform.IAAffinity;
import com.intel.analytics.bigdl.mkl.hardware.platform.linux.LinuxAffinity;

public enum Affinity {
    INSTANCE;

    private static final IAAffinity IMPL;

    static {
        IMPL = new LinuxAffinity(CpuInfo.INSTANCE.getLogicalProcessorCount());
    }

    public static int setAffinity() {
        return IMPL.setAffinity();
    }

    public static void setAffinity(int cpuId) {
        int[] set = {cpuId};
        IMPL.setAffinity(set);
    }

    public static void setAffinity(int[] cpuIds) {
        IMPL.setAffinity(cpuIds);
    }

    public static int[] getAffinity() {
        return IMPL.getAffinity();
    }

    public static void setOmpAffinity() {
        IMPL.setOmpAffinity();
    }
}
