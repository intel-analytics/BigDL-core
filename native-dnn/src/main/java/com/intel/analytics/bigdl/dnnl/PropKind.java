package com.intel.analytics.bigdl.dnnl;

public class PropKind {
    public static final int Undef = 0;
    public static final int ForwardTraining = 64;
    public static final int ForwardInference = 96;
    public static final int ForwardScoring = 96;
    public static final int Forward = 64;
    public static final int Backward = 128;
    public static final int BackwardData = 160;
    public static final int BackwardWeights = 192;
    public static final int BackwardBias = 193;
}
