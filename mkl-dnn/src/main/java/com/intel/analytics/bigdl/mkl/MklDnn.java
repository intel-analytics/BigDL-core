/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.mkl;

public class MklDnn {
    private static boolean _isLoaded = false;

    static {
        try {
            Loader loader = new Loader();
            loader.init();

            _isLoaded = true;
        } catch (Exception e) {
            _isLoaded = false;

            e.printStackTrace();
            throw new RuntimeException("Failed to load MklDnn");
        }
    }

    public static class EngineType {
        public static final int any = 0;
        public static final int cpu = 1;
    }

    public static class StreamType {
        public static final int any           = 0;
        public static final int eager         = 1;
        public static final int lazy          = 2;
    }

    public static class DataType {
        public static final int undef       = 0;
        public static final int f32         = 1;
        public static final int s32         = 2;
        public static final int s16         = 4;
        public static final int s8          = 5;
        public static final int u8          = 6;
    }

    public static class MemoryFormat {
        public static final int undef         = 0;
        public static final int any           = 1;
        public static final int blocked       = 2;
        public static final int x             = 3;
        public static final int nc            = 4;
        public static final int nchw          = 5;
        public static final int nhwc          = 6;
        public static final int chwn          = 7;
        public static final int nChw8c        = 8;
        public static final int nChw16c       = 9;
        public static final int oi            = 10;
        public static final int io            = 11;
        public static final int oihw          = 12;
        public static final int ihwo          = 13;
        public static final int hwio          = 14;
        public static final int OIhw8i8o      = 15;
        public static final int OIhw16i16o    = 16;
        public static final int OIhw8i16o2i   = 17;
        public static final int OIhw8o16i2o   = 18;
        public static final int OIhw8o8i      = 19;
        public static final int OIhw16o16i    = 20;
        public static final int IOhw16o16i    = 21;
        public static final int Oihw8o        = 22;
        public static final int Oihw16o       = 23;
        public static final int Ohwi8o        = 24;
        public static final int Ohwi16o       = 25;
        public static final int OhIw16o4i     = 26;
        public static final int goihw         = 27;
        public static final int gOIhw8i8o     = 28;
        public static final int gOIhw16i16o   = 29;
        public static final int gOIhw8i16o2i  = 30;
        public static final int gOIhw8o16i2o  = 31;
        public static final int gOIhw8o8i     = 32;
        public static final int gOIhw16o16i   = 33;
        public static final int gIOhw16o16i   = 34;
        public static final int gOihw8o       = 35;
        public static final int gOihw16o      = 36;
        public static final int gOhwi8o       = 37;
        public static final int gOhwi16o      = 38;
        public static final int gOhIw16o4i    = 39;
        public static final int oIhw8i        = nChw8c;
        public static final int oIhw16i       = nChw16c;
    }

    public static class PropKind {
        public static final int undef            = 0;
        public static final int forwardTraining  = 64;
        public static final int forwardInference = 96;
        public static final int forwardScoring   = forwardInference;
        public static final int forward          = forwardTraining;
        public static final int backward         = 128;
        public static final int backwardData     = 160;
        public static final int backwardWeights  = 192;
        public static final int backwardBias     = 193;
    }

    public static class AlgKind {
        public static final int convolutionDirect        = 1;
        public static final int convolutionWinograd      = 2;
        public static final int eltwiseRelu              = 8;
        public static final int eltwiseTanh              = 9;
        public static final int eltwiseElu               = 10;
        public static final int eltwiseSquare            = 11;
        public static final int eltwiseAbs               = 12;
        public static final int eltwiseSqrt              = 13;
        public static final int eltwiseLinear            = 14;
        public static final int eltwiseBoundedRelu       = 15;
        public static final int eltwisesoftRelu          = 16;
        public static final int eltwiselogistic          = 17;
        public static final int poolingMax               = 34;
        public static final int poolingAvgIncludePadding = 40;
        public static final int poolingAvgExcludePadding = 41;
        public static final int poolingAvg               = poolingAvgExcludePadding;
        public static final int lrnAcrossChannels        = 65;
        public static final int lrnWithinChannel         = 66;
    }

    public static class PaddingKind {
        public static final int mkldnnPaddingZero        = 0;
    }



    public static class Query {
        public static final int undef                      = 0;
        public static final int engine                     = 1;
        public static final int primitive_kind             = 2;
        public static final int num_of_inputs_s32          = 3;
        public static final int num_of_outputs_s32         = 4;
        public static final int time_estimate_f64          = 5;
        public static final int memory_consumption_s64     = 6;
        public static final int impl_info_str              = 7;
        /* memory and op descriptor section */
        public static final int some_d                     = 64;
        public static final int memory_d                   = 65;
        public static final int convolution_d              = 66;
        public static final int eltwise_d                  = 67;
        public static final int relu_d                     = eltwise_d;
        public static final int softmax_d                  = 68;
        public static final int pooling_d                  = 69;
        public static final int lrn_d                      = 70;
        public static final int batch_normalization_d      = 71;
        public static final int inner_product_d            = 72;
        public static final int convolution_relu_d         = 73;
        /* (memory) primitive descriptor section */
        public static final int some_pd                    = 128;
        public static final int input_pd                   = 129;
        public static final int output_pd                  = 130;
        public static final int src_pd                     = 131;
        public static final int diff_src_pd                = 132;
        public static final int weights_pd                 = 133;
        public static final int diff_weights_pd            = 134;
        public static final int dst_pd                     = 135;
        public static final int diff_dst_pd                = 136;
        public static final int workspace_pd               = 137;
    }

    public static class BatchNormFlag {
        /** Use global statistics
         *
         * If specified
         *  - on forward propagation use mean and variance provided by user (input)
         *  - on backward propagation reduces the amount of computations, since
         *    mean and variance are considered as constants
         *
         *  If not specified:
         *   - on forward propagation mean and variance are computed and stored in
         *     output
         *   - on backward propagation compute full derivative wrt to data
         */
        // TODO 0x1 original value is 0x1U
        public static final long mkldnn_use_global_stats = 0x1;
        /** Use scale and shift parameters
         *
         * If specified:
         *  - on forward propagation use scale and shift (aka scale and bias) for
         *    the batch normalization results
         *  - on backward propagation (for prop_kind == #mkldnn_backward) compute
         *    diff wrt to scale and shift (hence one extra output used)
         *
         * If no specified:
         *  - on backward propagation prop_kind == #mkldnn_backward_data has the
         *    same behavior as prop_kind == #mkldnn_backward
         */
        public static final long mkldnn_use_scaleshift = 0x2;
        /** Omit statistics
         *
         * @warning: deprecated, use #mkldnn_use_global_stats instead
         *
         * For time being had an affect on backward propagation only which allowed
         * skipping some computations (the same semantics as
         * #mkldnn_use_global_stats)
         */
        public static final long mkldnn_omit_stats = mkldnn_use_global_stats;
    }

    public static boolean isLoaded() {
        return _isLoaded;
    }

    public native static long EngineCreate(int id, int index);
    public native static void EngineDestroy(long engine);

    public native static long StreamCreate(int streamKind);
    public native static void StreamSubmit(long stream, int length, long[] primitives);

    public native static long StreamWait(long loc, int block);
    public native static long StreamRerun(long stream);
    public native static void StreamDestroy(long loc);

    public native static long MemoryDescInit(int ndims, int[] dims,
                                             int dataType, int dataFormat);
    public native static long MemoryPrimitiveDescCreate(long desc, long engine);

    public native static long MemoryGetDataHandle(long memory);
    public native static long MemorySetDataHandle(long memory, float[] data, int offset);
    public native static void MemoryReleaseDataHandle(float[] data, long ptr);

    public native static long PrimitiveCreate0(long desc);
    public native static long PrimitiveCreate2(long desc,
                                               long[] inputs,
                                               int[] indexes,
                                               int inputLen,
                                               long[] outputs,
                                               int outputLen);

    public native static long PrimitiveDescCreate(long opDesc, long engine,
                                                  long hingForwardPrimitiveDesc);
    public native static void PrimitiveDescDestroy(long desc);
    public native static void PrimitiveDestroy(long primitive);

    public native static long EltwiseForwardDescInit(int propKind, int algKind,
                                                     long srcDesc, float alpha, float beta);
    public native static long EltwiseBackwardDescInit(int algKind, long diffDataDesc,
                                                      long dataDesc, float alpha, float beta);

    public native static long LinearForwardDescInit(int propKind,
                                                    long srcMemDesc,
                                                    long weightMemDesc,
                                                    long biasMemDesc,
                                                    long dstMemDesc);

    public native static long LinearBackwardDataDescInit(long diffSrcMemDesc,
                                                         long weightMemDesc,
                                                         long diffDstMemDesc);

    public native static long LinearBackwardWeightsDescInit(long srcMemDesc,
                                                            long diffWeightMemDesc,
                                                            long diffBiasMemDesc,
                                                            long diffDstMemDesc);

    public native static long BatchNormForwardDescInit(int propKind,
                                                       long srcMemDesc,
                                                       float epsilon,
                                                       long flags);

    public native static long BatchNormBackwardDescInit(int prop_kind,
                                                        long diffDstMemDesc,
                                                        long srcMemDesc,
                                                        float epsilon,
                                                        long flags);

    public native static long SoftMaxForwardDescInit(int prop_kind,
                                                    long dataDesc,
                                                    int axis);

    public native static long ConvForwardDescInit(int prop_kind, int alg_kind,
                                                  long src_desc, long weights_desc,
                                                  long bias_desc, long dst_desc,
                                                  int[] strides, int[] padding_l,
                                                  int[] padding_r, int padding_kind);

    public native static long ConvBackwardWeightsDescInit(int alg_kind, long src_desc,
                                                          long diff_weights_desc,
                                                          long diff_bias_desc,
                                                          long diff_dst_desc, int[] strides,
                                                          int[] padding_l, int[] padding_r,
                                                          int padding_kind);

    public native static long ConvBackwardDataDescInit(int alg_kind, long diff_src_desc,
                                                       long weights_desc, long diff_dst_desc,
                                                       int[] strides, int[] padding_l,
                                                       int[] padding_r, int padding_kind);

    public native static long PoolingForwardDescInit(int prop_kind, int alg_kind,
                                                     long src_desc, long dst_desc,
                                                     int[] strides, int[] kernel,
                                                     int[] padding_l, int[] padding_r,
                                                     int padding_kind);

    public native static long PoolingBackwardDescInit(int alg_kind, long diff_src_desc,
                                                      long diff_dst_desc, int[] strides,
                                                      int[] kernel, int[] padding_l,
                                                      int[] padding_r, int padding_kind);

    public native static long ReorderPrimitiveDescCreate(long input, long output);

    public native static int MemoryPrimitiveDescEqual(long lhs, long rhs);

    public native static long PrimitiveGetPrimitiveDesc(long primitive);

    public native static long PrimitiveDescQueryPd(long primitive, int what, int index);

    public native static long PrimitiveDescQueryMemory(long primitive_desc);

    public native static long PrimitiveDescGetSize(long primitive_desc);

    public native static long LRNForwardDescInit(int prop_kind, int alg_kind, long data_desc,
                                                int local_size, float alpha, float beta, float k);

    public native static long LRNBackwardDescInit(int alg_kind, long diff_data_desc, long data_desc,
                                                 int local_size, float alpha, float beta, float k);
}
