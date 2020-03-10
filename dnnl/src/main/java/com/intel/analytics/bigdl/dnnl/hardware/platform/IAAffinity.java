package com.intel.analytics.bigdl.dnnl.hardware.platform;

import java.util.List;
import java.util.Map;

public interface IAAffinity {
    void setAffinity();
    void setAffinity(int[] sets);
    void resetAffinity();
    void setOmpAffinity();
    int[] getAffinity();
    int[] getOmpAffinity();
    Map<Integer, List<Long>> stats();
}
