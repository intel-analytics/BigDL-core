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

import java.nio.FloatBuffer;

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

    public static class PaddingKind {
        public static final int mkldnnPaddingZero        = 0;
    }

    public static class BatchNormFlag {
        public static final long mkldnn_use_global_stats = 0x1;
        public static final long mkldnn_use_scaleshift = 0x2;
        public static final long mkldnn_omit_stats = mkldnn_use_global_stats;
    }

    public static boolean isLoaded() {
        return _isLoaded;
    }
    public native static void setNumThreads(int num);

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
    public native static long PrimitiveDescCreateV2(long opDesc, long attr, long engine,
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

    public native static long DilatedConvForwardDescInit(int prop_kind, int alg_kind,
                                                  long src_desc, long weights_desc,
                                                  long bias_desc, long dst_desc,
                                                  int[] strides, int[] dilates,
                                                  int[] padding_l, int[] padding_r,
                                                  int padding_kind);

    public native static long ConvBackwardWeightsDescInit(int alg_kind, long src_desc,
                                                          long diff_weights_desc,
                                                          long diff_bias_desc,
                                                          long diff_dst_desc, int[] strides,
                                                          int[] padding_l, int[] padding_r,
                                                          int padding_kind);

    public native static long DilatedConvBackwardWeightsDescInit(int alg_kind, long src_desc,
                                                          long diff_weights_desc,
                                                          long diff_bias_desc,
                                                          long diff_dst_desc,
                                                          int[] strides, int[] dilates,
                                                          int[] padding_l, int[] padding_r,
                                                          int padding_kind);

    public native static long ConvBackwardDataDescInit(int alg_kind, long diff_src_desc,
                                                       long weights_desc, long diff_dst_desc,
                                                       int[] strides, int[] padding_l,
                                                       int[] padding_r, int padding_kind);

    public native static long DilatedConvBackwardDataDescInit(int alg_kind, long diff_src_desc,
                                                       long weights_desc, long diff_dst_desc,
                                                       int[] strides,
                                                       int[] padding_l, int[] dilates,
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
    public native static long ReorderPrimitiveDescCreateV2(long input, long output, long attr);

    public native static int MemoryPrimitiveDescEqual(long lhs, long rhs);

    public native static long PrimitiveGetPrimitiveDesc(long primitive);

    public native static long PrimitiveDescQueryPd(long primitive, int what, int index);

    public native static long PrimitiveDescQueryMemory(long primitive_desc);

    public native static long PrimitiveDescGetSize(long primitive_desc);

    public native static long LRNForwardDescInit(int prop_kind, int alg_kind, long data_desc,
                                                int local_size, float alpha, float beta, float k);

    public native static long LRNBackwardDescInit(int alg_kind, long diff_data_desc, long data_desc,
                                                 int local_size, float alpha, float beta, float k);

    public native static long RNNCellDescInit(int kind, int f, int flags, float alpha, float clipping);

    public native static int RNNCellGetGatesCount(long rnn_cell_desc);

    public native static int RNNCellGetStatesCount(long rnn_cell_desc);

    public native static long RNNForwardDescInit(int prop_kind, long rnn_cell_desc,
                                                 int direction, long src_layer_desc,
                                                 long src_iter_desc, long weights_layer_desc,
                                                 long weights_iter_desc, long bias_desc,
                                                 long dst_layer_desc, long dst_iter_desc);

    public native static long RNNBackwardDescInit(int prop_kind, long rnn_cell_desc,
                                                  int direction, long src_layer_desc,
                                                  long src_iter_desc, long weights_layer_desc,
                                                  long weights_iter_desc, long bias_desc,
                                                  long dst_layer_desc, long dst_iter_desc,
                                                  long diff_src_layer_desc, long diff_src_iter_desc,
                                                  long diff_weights_layer_desc, long diff_weights_iter_desc,
                                                  long diff_bias_desc, long diff_dst_layer_desc,
                                                  long diff_dst_iter_desc);

    // get format from memory desc
    public native static int getFormat(long memoryDesc);

    // get size from memory primitive desc
    public native static int getSize(long memoryPrimDesc);

    // direct buffer
    public native static void copyFloatBuffer2Array(FloatBuffer buffer, int bufferOffset,
                                                    float[] array, int arrayOffset, int length);
    public native static void copyArray2FloatBuffer(FloatBuffer buffer, int bufferOffset,
                                                    float[] array, int arrayOffset, int length);
    public native static void fillFloatBuffer(FloatBuffer buffer, int bufferOffset,
                                              float value, int length);

    public native static long MemoryGetDataHandleOfArray(float[] array);

    public native static void MemorySetDataHandleWithBuffer(long primitive,
                                                            long array,
                                                            int offset,
                                                            int length,
                                                            FloatBuffer buffer,
                                                            int position);

    public native static void MemorySetDataHandleWithPtr(long primitive,
                                                         long array,
                                                         int offset,
                                                         int length,
                                                         long buffer,
                                                         int position);
    public native static void copyPtr2Array(long buffer, int bufferOffset,
                                            float[] array, int arrayOffset, int length);
    public native static long MemoryAlignedMalloc(int capacity, int align);
    public native static void MemoryAlignedFree(long ptr);

    public native static long ConcatPrimitiveDescCreate(long output_desc, int n,
                                                        int concat_dimension,
                                                        long[] input_pds);

    public native static long ViewPrimitiveDescCreate(long memory_primitive_desc,
                                                      int[] dims,
                                                      int[] offsets);

    public native static long SumPrimitiveDescCreate(long output_mem_desc, int n, float[] scales,
                                                     long[] input_pds);


    public native static long ConcatPrimitive(long output_desc, int n,
                                              int concat_dimension,
                                              long[] input_pds, long engine,
                                              long input1_memory,
                                              long input2_memory,
                                              long dst_memory);

    public native static long PrimitiveCreateNew(long concat_desc,
                                              long input1_memory,
                                              long input2_memory,
                                              long dst_memory);

    public native static void FreeUnuse(long dnn_desc);

    public native static void FreeBatchNormDescInit(long bn_desc);
    public native static void FreeConcatDescInit(long concat_desc);
    public native static void FreeViewDescInit(long view_primitive_desc);
    public native static void FreeConvDescInit(long conv_desc);
    public native static void FreeEltwiseDescInit(long relu_desc);
    public native static void FreeLinearDescInit(long ip_desc);
    public native static void FreeLRNDescInit(long lrn_desc);
    public native static void FreeMemoryDescInit(long memory_desc);
    public native static void FreePoolDescInit(long pool_desc);
    public native static void FreeSoftMaxDescInit(long sm_desc);
    public native static void FreeRNNCellDescInit(long rnn_cell_desc);
    public native static void FreeRNNDescInit(long rnn_desc);

    // post ops
    public native static long CreatePostOps();
    public native static void DestroyPostOps(long postOps);
    public native static void PostOpsAppendEltwise(long postOps, float scale,
                                                   int kind, float alpha, float beta);
    public native static void PostOpsAppendSum(long postOps, float scale);
    public native static void AttrSetPostOps(long attr, long postOps);

    // attr
    public native static long CreateAttr();
    public native static void DestroyAttr(long attr);
    public native static void AttrSetIntOutputRoundMode(long attr, int roundMode);
    public native static void AttrSetOutputScales(long attr, int count, int mask, float[] scales);
}
