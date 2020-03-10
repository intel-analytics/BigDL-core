package com.intel.analytics.bigdl.dnnl.hardware.platform.linux;

import com.intel.analytics.bigdl.dnnl.hardware.platform.IANativeUtils;

public class LinuxNativeUtils implements IANativeUtils {
    private native static int getTaskId0();
    private native static int getPid0();

    public int getTaskId() {
        return getTaskId0();
    }

    public int getPid() {
        return getPid0();
    }
}
