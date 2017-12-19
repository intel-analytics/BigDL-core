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
        final int any = 0;
        final int cpu = 1;
    }

    public static class StreamType {
        final static int any           = 0;
        final static int eager         = 0;
        final static int lazy          = 0;
    }

    public static class DataType {
        final static int undef       = 0;
        final static int f32         = 1;
        final static int s32         = 2;
        final static int s16         = 4;
        final static int s8          = 5;
        final static int u8          = 6;
    }

    public static class MemoryFormat {
        final static int undef         = 0;
        final static int any           = 1;
        final static int blocked       = 2;
        final static int x             = 3;
        final static int nc            = 4;
        final static int nchw          = 5;
        final static int nhwc          = 6;
        final static int chwn          = 7;
        final static int nChw8c        = 8;
        final static int nChw16c       = 9;
        final static int oi            = 10;
        final static int io            = 11;
        final static int oihw          = 12;
        final static int ihwo          = 13;
        final static int hwio          = 14;
        final static int OIhw8i8o      = 15;
        final static int OIhw16i16o    = 16;
        final static int OIhw8i16o2i   = 17;
        final static int OIhw8o16i2o   = 18;
        final static int OIhw8o8i      = 19;
        final static int OIhw16o16i    = 20;
        final static int IOhw16o16i    = 21;
        final static int Oihw8o        = 22;
        final static int Oihw16o       = 23;
        final static int Ohwi8o        = 24;
        final static int Ohwi16o       = 25;
        final static int OhIw16o4i     = 26;
        final static int goihw         = 27;
        final static int gOIhw8i8o     = 28;
        final static int gOIhw16i16o   = 29;
        final static int gOIhw8i16o2i  = 30;
        final static int gOIhw8o16i2o  = 31;
        final static int gOIhw8o8i     = 32;
        final static int gOIhw16o16i   = 33;
        final static int gIOhw16o16i   = 34;
        final static int gOihw8o       = 35;
        final static int gOihw16o      = 36;
        final static int gOhwi8o       = 37;
        final static int gOhwi16o      = 38;
        final static int gOhIw16o4i    = 39;
        final static int oIhw8i        = nChw8c;
        final static int oIhw16i       = nChw16c;
    }

    public class PropKind {
        final static int undef            = 0;
        final static int forwardTraining  = 64;
        final static int forwardInference = 96;
        final static int forwardScoring   = forwardInference;
        final static int forward          = forwardTraining;
        final static int backward         = 128;
        final static int backwardData     = 160;
        final static int backwardWeights  = 192;
        final static int backwardBias     = 193;
    }

    public class AlgKind {
        final static int convolutionDirect        = 1;
        final static int convolutionWinograd      = 2;
        final static int eltwiseRelu              = 8;
        final static int eltwiseTanh              = 9;
        final static int eltwiseElu               = 10;
        final static int eltwiseSquare            = 11;
        final static int eltwiseAbs               = 12;
        final static int eltwiseSqrt              = 13;
        final static int eltwiseLinear            = 14;
        final static int eltwiseBoundedRelu       = 15;
        final static int eltwisesoftRelu          = 16;
        final static int eltwiselogistic          = 17;
        final static int poolingMax               = 34;
        final static int poolingAvgIncludePadding = 40;
        final static int poolingAvgExcludePadding = 41;
        final static int poolingAvg               = poolingAvgExcludePadding;
        final static int lrnAcrossChannels        = 65;
        final static int lrnWithinChannel         = 66;
    }

    public static boolean isLoaded() {
        return _isLoaded;
    }

    public native static long EngineCreate(long id, long index);
    public native static void EngineDestroy(long engine);

    public native static long StreamCreate(int streamKind);
    public native static long StreamSubmit(long loc, int block, long[] primitives);
    public native static long StreamWait(long loc, int block);
    public native static void StreamDestroy(long loc);

    public native static long MemoryDescInit(int ndims, int[] dims,
                                             int dataType, int dataFormat);
    public native static long MemoryPrimitiveDescCreate(long desc, int engine);

    public native static long MemoryGetDataHandle(long memory);
    public native static void MemorySetDataHandle(long memory, long data);

    public native static long PrimitiveCreate(long desc, long[] inputs, long[] outputs);
    public native static long PrimitiveDescCreate(long opDesc, int engine,
                                                  long hingForwardPrimitiveDesc);
    public native static void PrimitiveDescDestroy(long desc);
    public native static void PrimitiveDestroy(long primitive);

    public native static long EltwiseForwardDescInit(int propKind, int algKind,
                                                     long srcDesc, float alpha, float beta);
    public native static long EltwiseBackwardDescInit(int algKind, long diffDataDesc,
                                                      long dataDesc, float alpha, float beta);
}
