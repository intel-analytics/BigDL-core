package com.intel.analytics.bigdl.mkl.hardware;

import com.intel.analytics.bigdl.mkl.MklDnn;
import com.intel.analytics.bigdl.mkl.hardware.platform.IACpuInfo;
import com.intel.analytics.bigdl.mkl.hardware.platform.linux.LinuxCpuInfo;

public enum CpuInfo {
    INSTANCE;

    private final static IACpuInfo IMPL;

    static {
        MklDnn.isLoaded();
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
