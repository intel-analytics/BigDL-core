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

public class Memory {
    static {
        MklDnn.isLoaded();
    }

    public static class Format {
        public static final int format_undef = 0;
        public static final int any = 1;
        public static final int auto = any;
        public static final int blocked = 2;
        public static final int x = 3;
        public static final int nc = 4;
        public static final int nchw = 5;
        public static final int nhwc = 6;
        public static final int chwn = 7;
        public static final int nChw8c = 8;
        public static final int nChw16c = 9;
        public static final int ncdhw = 10;
        public static final int ndhwc = 11;
        public static final int nCdhw16c = 12;
        public static final int oi = 13;
        public static final int io = 14;
        public static final int oihw = 15;
        public static final int ihwo = 16;
        public static final int hwio = 17;
        public static final int dhwio = 18;
        public static final int oidhw = 19;
        public static final int OIdhw16i16o = 20;
        public static final int OIdhw16o16i = 21;
        public static final int Oidhw16o = 22;
        public static final int Odhwi16o = 23;
        public static final int OIhw8i8o = 24;
        public static final int OIhw16i16o = 25;
        public static final int OIhw4i16o4i = 26;
        public static final int OIhw8i16o2i = 27;
        public static final int OIhw8o16i2o = 28;
        public static final int OIhw8o8i = 29;
        public static final int OIhw16o16i = 30;
        public static final int IOhw16o16i = 31;
        public static final int Oihw8o = 32;
        public static final int Oihw16o = 33;
        public static final int Ohwi8o = 34;
        public static final int Ohwi16o = 35;
        public static final int OhIw16o4i = 36;
        public static final int goihw = 37;
        public static final int hwigo = 38;
        public static final int gOIhw8i8o = 39;
        public static final int gOIhw16i16o = 40;
        public static final int gOIhw4i16o4i = 41;
        public static final int gOIhw8i16o2i = 42;
        public static final int gOIhw8o16i2o = 43;
        public static final int gOIhw8o8i = 44;
        public static final int gOIhw16o16i = 45;
        public static final int gIOhw16o16i = 46;
        public static final int gOihw8o = 47;
        public static final int gOihw16o = 48;
        public static final int gOhwi8o = 49;
        public static final int gOhwi16o = 50;
        public static final int Goihw8g = 51;
        public static final int Goihw16g = 52;
        public static final int gOhIw16o4i = 53;
        public static final int goidhw = 54;
        public static final int gOIdhw16i16o = 55;
        public static final int gOIdhw16o16i = 56;
        public static final int gOidhw16o = 57;
        public static final int gOdhwi16o = 58;
        public static final int ntc = 59;
        public static final int tnc = 60;
        public static final int ldsnc = 61;
        public static final int ldigo = 62;
        public static final int ldigo_p = 63;
        public static final int ldgoi = 64;
        public static final int ldgoi_p = 65;
        public static final int ldgo = 66;
        public static final int wino_fmt = 67;
        public static final int format_last = 68;
        public static final int oIhw8i = 8;
        public static final int oIhw16i = 9;
    }

    public native static long SetDataHandle(long memoryPrimitive, long data, int offset);
    public native static long Zero(long data, int length, int elementSize);
    // TODO use override methods
    public native static long CopyPtr2Ptr(long src, int srcOffset, long dst, int dstOffset,
                                          int length, int elementSize);
    public native static long CopyArray2Ptr(float[] src, int srcOffset, long dst, int dstOffset,
                                            int length, int elementSize);
    public native static long CopyPtr2Array(long src, int srcOffset, float[] dst, int dstOffset,
                                            int length, int elementSize);
    public native static long AlignedMalloc(int capacity, int size);
    public native static void AlignedFree(long ptr);

    public native static void SAdd(int n, long aPtr, int aOffset, long bPtr, int bOffset,
                                    long yPtr, int yOffset);

    public native static void Scale(int n, float scaleFactor, long from, long to);
    public native static void Axpby(int n, float a, long x, float b, long y);
    public native static void Set(long data, float value, int length, int elementSize);

    public native static int[] GetShape(long desc);
    public native static int GetLayout(long desc);
}
