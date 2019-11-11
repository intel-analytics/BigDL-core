package com.intel.analytics.bigdl.dnnl.hardware.platform;

public interface IACpuInfo {
    int getPhysicalProcessorCount();
    int getLogicalProcessorCount();
    int getSocketsCount();
    boolean isEnableHyperThreading();
}
