package com.intel.analytics.bigdl.mkl;

public class AlgKind {
    public static final int Undef = 0;
    public static final int ConvolutionDirect = 1;
    public static final int ConvolutionWinograd = 2;
    public static final int EltwiseRelu = 8;
    public static final int EltwiseTanh = 9;
    public static final int EltwiseElu = 10;
    public static final int EltwiseSquare = 11;
    public static final int EltwiseAbs = 12;
    public static final int EltwiseSqrt = 13;
    public static final int EltwiseLinear = 14;
    public static final int EltwiseBoundedRelu = 15;
    public static final int EltwiseSoftRelu = 16;
    public static final int EltwiseLogistic = 17;
    public static final int PoolingMax = 34;
    public static final int PoolingAvgIncludePadding = 40;
    public static final int PoolingAvgExcludePadding = 41;
    public static final int PoolingAvg = 41;
    public static final int LrnAcrossChannels = 65;
    public static final int LrnWithinChannel = 66;
    public static final int DeconvolutionDirect = 71;
    public static final int DeconvolutionWinograd = 72;
    public static final int VanillaRnn = 80;
    public static final int VanillaLstm = 81;
    public static final int VanillaGru = 82;
}

