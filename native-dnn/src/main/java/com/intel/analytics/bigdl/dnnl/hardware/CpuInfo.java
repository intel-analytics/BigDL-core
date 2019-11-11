package com.intel.analytics.bigdl.dnnl.hardware;

import com.intel.analytics.bigdl.dnnl.DNNL;
import com.intel.analytics.bigdl.dnnl.hardware.platform.IACpuInfo;
import com.intel.analytics.bigdl.dnnl.hardware.platform.linux.LinuxCpuInfo;

public enum CpuInfo {
    INSTANCE;

    private final static IACpuInfo IMPL;

    static {
        DNNL.isLoaded();
        IMPL = new LinuxCpuInfo();
    }

    public static int getPhysicalProcessorCount() {
        return IMPL.getPhysicalProcessorCount();
    }

    public static int getLogicalProcessorCount() {
        return IMPL.getLogicalProcessorCount();
    }

    public static int getSocketsCount() {
        return IMPL.getSocketsCount();
    }

    public static boolean isEnableHyperThreading() {
        return IMPL.isEnableHyperThreading();
    }
}
