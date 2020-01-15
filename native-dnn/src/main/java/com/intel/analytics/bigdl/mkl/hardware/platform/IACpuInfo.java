package com.intel.analytics.bigdl.mkl.hardware.platform;

public interface IACpuInfo {
    int getPhysicalProcessorCount();
    int getLogicalProcessorCount();
    int getSocketsCount();
    boolean isEnableHyperThreading();
}
