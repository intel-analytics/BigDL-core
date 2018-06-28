package com.intel.analytics.bigdl.mkl.hardware;

import com.intel.analytics.bigdl.mkl.hardware.platform.IANativeUtils;
import com.intel.analytics.bigdl.mkl.hardware.platform.linux.LinuxNativeUtils;

public enum NativeUtils {
    INSTANCE;

    private final static IANativeUtils IMPL;

    static {
        IMPL = new LinuxNativeUtils();
    }

    public static int getPid() {
        return IMPL.getPid();
    }

    public static int getTaskId() {
        return IMPL.getTaskId();
    }
}
