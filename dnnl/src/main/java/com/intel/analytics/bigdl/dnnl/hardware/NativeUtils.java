package com.intel.analytics.bigdl.dnnl.hardware;

import com.intel.analytics.bigdl.dnnl.DNNL;
import com.intel.analytics.bigdl.dnnl.hardware.platform.IANativeUtils;
import com.intel.analytics.bigdl.dnnl.hardware.platform.linux.LinuxNativeUtils;

public enum NativeUtils {
    INSTANCE;

    private final static IANativeUtils IMPL;

    static {
        DNNL.isLoaded();
        IMPL = new LinuxNativeUtils();
    }

    public static int getPid() {
        return IMPL.getPid();
    }

    public static int getTaskId() {
        return IMPL.getTaskId();
    }
}
