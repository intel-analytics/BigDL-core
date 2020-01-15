package com.intel.analytics.bigdl.dnnl;

public class AlgKind {
    public static final int AlgKindUndef = 0;
    public static final int ConvolutionDirect = 1;
    public static final int ConvolutionWinograd = 2;
    public static final int ConvolutionAuto = 3;
    public static final int DeconvolutionDirect = 10;
    public static final int DeconvolutionWinograd = 11;
    public static final int EltwiseRelu = 31;
    public static final int EltwiseTanh = 47;
    public static final int EltwiseElu = 63;
    public static final int EltwiseSquare = 79;
    public static final int EltwiseAbs = 95;
    public static final int EltwiseSqrt = 111;
    public static final int EltwiseLinear = 127;
    public static final int EltwiseBoundedRelu = 143;
    public static final int EltwiseSoftRelu = 159;
    public static final int EltwiseLogistic = 175;
    public static final int EltwiseExp = 191;
    public static final int EltwiseGelu = 207;
    public static final int EltwiseSwish = 223;
    public static final int PoolingMax = 511;
    public static final int PoolingAvgIncludePadding = 767;
    public static final int PoolingAvgExcludePadding = 1023;
    public static final int PoolingAvg = 1023;
    public static final int LrnAcrossChannels = 2815;
    public static final int LrnWithinChannel = 3071;
    public static final int VanillaRnn = 8191;
    public static final int VanillaLstm = 12287;
    public static final int VanillaGru = 16383;
    public static final int LbrGru = 20479;
    public static final int BinaryAdd = 131056;
    public static final int BinaryMul = 131057;
}
