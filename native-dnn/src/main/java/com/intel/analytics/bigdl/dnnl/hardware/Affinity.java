package com.intel.analytics.bigdl.dnnl.hardware;

import com.intel.analytics.bigdl.dnnl.DNNL;
import com.intel.analytics.bigdl.dnnl.hardware.platform.IAAffinity;
import com.intel.analytics.bigdl.dnnl.hardware.platform.linux.LinuxAffinity;

import java.util.List;
import java.util.Map;

public enum Affinity {
    INSTANCE;

    private static final IAAffinity IMPL;

    static {
        DNNL.isLoaded();
        IMPL = new LinuxAffinity(CpuInfo.INSTANCE.getLogicalProcessorCount());
    }

    public static void setAffinity() {
        IMPL.setAffinity();
    }

    public static void setAffinity(int coreId) {
        int[] set = {coreId};
        IMPL.setAffinity(set);
    }

    public static void setAffinity(int[] coreIds) {
        IMPL.setAffinity(coreIds);
    }

    public static void resetAffinity() {
        IMPL.resetAffinity();
    }

    public static int[] getAffinity() {
        return IMPL.getAffinity();
    }

    public static void setOmpAffinity() {
        IMPL.setOmpAffinity();
    }

    public static int[] getOmpAffinity() {
        return IMPL.getOmpAffinity();
    }

    public static Map<Integer, List<Long>> stats() {
        return IMPL.stats();
    }
}
