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

    public static boolean isLoaded() {
        return _isLoaded;
    }

    public native static long EngineCreate(long id, long index);
    public native static void EngineDestroy(long engine);

    public native static long StreamCreate(int streamKind);
    public native static long StreamSubmit(long loc, int block, long[] primitives, int length);
    public native static long StreamWait(long loc, int block);
    public native static void StreamDestroy(long loc);

    public native static long MemoryDescInit(int ndims, int[] dims,
                                             int dataType, int dataFormat);
    public native static long MemoryPrimitiveDescCreate(long desc, long engine);

    public native static long MemoryGetDataHandle(long memory);
    public native static long MemorySetDataHandle(long memory, float[] data);
    public native static void MemoryReleaseDataHandle(float[] data, long ptr);

    public native static long PrimitiveCreate(long desc, long[] inputs, long[] outputs);
    public native static long PrimitiveDescCreate(long opDesc, long engine,
                                                  long hingForwardPrimitiveDesc);
    public native static void PrimitiveDescDestroy(long desc);
    public native static void PrimitiveDestroy(long primitive);

    public native static long PrimitiveCreateForSubmit(long desc, long[] inputs, int length1, long[] outputs, int length2);


    public native static long EltwiseForwardDescInit(int propKind, int algKind,
                                                     long srcDesc, float alpha, float beta);
    public native static long EltwiseBackwardDescInit(int algKind, long diffDataDesc,
                                                      long dataDesc, float alpha, float beta);
}
