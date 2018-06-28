package com.intel.analytics.bigdl.mkl.hardware.platform;

public interface IAAffinity {
    int setAffinity();
    void setAffinity(int[] sets);
    void setOmpAffinity();
    int[] getAffinity();
}
