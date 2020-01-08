package com.intel.analytics.bigdl.dnnl;

public class ArgType {

  public static final int DNNL_ARG_UNDEFINE = -1;

  public static final int DNNL_ARG_SRC_0 = 1;
  public static final int DNNL_ARG_SRC = 1;
  public static final int DNNL_ARG_SRC_LAYER = 1;
  public static final int DNNL_ARG_FROM = 1;

  public static final int DNNL_ARG_SRC_1 = 2;
  public static final int DNNL_ARG_SRC_ITER = 2;

  public static final int DNNL_ARG_SRC_2 = 3;
  public static final int DNNL_ARG_SRC_ITER_C = 3;

  public static final int DNNL_ARG_DST_0 = 17;
  public static final int DNNL_ARG_DST = 17;
  public static final int DNNL_ARG_TO = 17;
  public static final int DNNL_ARG_DST_LAYER = 17;

  public static final int DNNL_ARG_DST_1 = 18;
  public static final int DNNL_ARG_DST_ITER = 18;

  public static final int DNNL_ARG_DST_2 = 19;
  public static final int DNNL_ARG_DST_ITER_C = 19;

  public static final int DNNL_ARG_WEIGHTS_0 = 33;
  public static final int DNNL_ARG_WEIGHTS = 33;
  public static final int DNNL_ARG_SCALE_SHIFT = 33;
  public static final int DNNL_ARG_WEIGHTS_LAYER = 33;

  public static final int DNNL_ARG_WEIGHTS_1 = 34;
  public static final int DNNL_ARG_WEIGHTS_ITER = 34;

  public static final int DNNL_ARG_BIAS = 41;

  public static final int DNNL_ARG_MEAN = 49;
  public static final int DNNL_ARG_VARIANCE = 50;

  public static final int DNNL_ARG_WORKSPACE = 64;
  public static final int DNNL_ARG_SCRATCHPAD = 80;

  public static final int DNNL_ARG_DIFF_SRC_0 = 129;
  public static final int DNNL_ARG_DIFF_SRC = 129;
  public static final int DNNL_ARG_DIFF_SRC_LAYER = 129;

  public static final int DNNL_ARG_DIFF_SRC_1 = 130;
  public static final int DNNL_ARG_DIFF_SRC_ITER = 130;

  public static final int DNNL_ARG_DIFF_SRC_2 = 131;
  public static final int DNNL_ARG_DIFF_SRC_ITER_C = 131;

  public static final int DNNL_ARG_DIFF_DST_0 = 145;
  public static final int DNNL_ARG_DIFF_DST = 145;
  public static final int DNNL_ARG_DIFF_DST_LAYER = 145;

  public static final int DNNL_ARG_DIFF_DST_1 = 146;
  public static final int DNNL_ARG_DIFF_DST_ITER = 146;

  public static final int DNNL_ARG_DIFF_DST_2 = 147;
  public static final int DNNL_ARG_DIFF_DST_ITER_C = 147;

  public static final int DNNL_ARG_DIFF_WEIGHTS_0 = 161;
  public static final int DNNL_ARG_DIFF_WEIGHTS = 161;
  public static final int DNNL_ARG_DIFF_SCALE_SHIFT = 161;
  public static final int DNNL_ARG_DIFF_WEIGHTS_LAYER = 161;

  public static final int DNNL_ARG_DIFF_WEIGHTS_1 = 162;
  public static final int DNNL_ARG_DIFF_WEIGHTS_ITER = 162;

  public static final int DNNL_ARG_DIFF_BIAS = 169;

  public static final int DNNL_ARG_MULTIPLE_SRC = 1024;
  public static final int DNNL_ARG_MULTIPLE_DST = 2048;

}

