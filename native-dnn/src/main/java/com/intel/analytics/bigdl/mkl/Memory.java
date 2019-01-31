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
      public static final int blocked = 2; 
      public static final int x = 3; 
      public static final int nc = 4; 
      public static final int ncw = 5; 
      public static final int nwc = 6; 
      public static final int nchw = 7; 
      public static final int nhwc = 8; 
      public static final int chwn = 9; 
      public static final int ncdhw = 10; 
      public static final int ndhwc = 11; 
      public static final int oi = 12; 
      public static final int io = 13; 
      public static final int oiw = 14; 
      public static final int wio = 15; 
      public static final int oihw = 16; 
      public static final int hwio = 17; 
      public static final int ihwo = 18; 
      public static final int oidhw = 19; 
      public static final int dhwio = 20; 
      public static final int goiw = 21; 
      public static final int goihw = 22; 
      public static final int hwigo = 23; 
      public static final int goidhw = 24; 
      public static final int ntc = 25; 
      public static final int tnc = 26; 
      public static final int ldsnc = 27; 
      public static final int ldigo = 28; 
      public static final int ldgoi = 29; 
      public static final int ldgo = 30; 
      public static final int nCw8c = 31; 
      public static final int nCw16c = 32; 
      public static final int nChw8c = 33; 
      public static final int nChw16c = 34; 
      public static final int nCdhw8c = 35; 
      public static final int nCdhw16c = 36; 
      public static final int Owi8o = 37; 
      public static final int OIw8i8o = 38; 
      public static final int OIw8o8i = 39; 
      public static final int OIw16i16o = 40; 
      public static final int OIw16o16i = 41; 
      public static final int Oiw16o = 42; 
      public static final int Owi16o = 43; 
      public static final int OIw8i16o2i = 44; 
      public static final int OIw8o16i2o = 45; 
      public static final int IOw16o16i = 46; 
      public static final int hwio_s8s8 = 47; 
      public static final int oIhw8i = 48; 
      public static final int oIhw16i = 49; 
      public static final int OIhw8i8o = 50; 
      public static final int OIhw16i16o = 51; 
      public static final int OIhw4i16o4i = 52; 
      public static final int OIhw4i16o4i_s8s8 = 53; 
      public static final int OIhw8i16o2i = 54; 
      public static final int OIhw8o16i2o = 55; 
      public static final int OIhw8o8i = 56; 
      public static final int OIhw16o16i = 57; 
      public static final int IOhw16o16i = 58; 
      public static final int Oihw8o = 59; 
      public static final int Oihw16o = 60; 
      public static final int Ohwi8o = 61; 
      public static final int Ohwi16o = 62; 
      public static final int OhIw16o4i = 63; 
      public static final int oIdhw8i = 64; 
      public static final int oIdhw16i = 65; 
      public static final int OIdhw8i8o = 66; 
      public static final int OIdhw8o8i = 67; 
      public static final int Odhwi8o = 68; 
      public static final int OIdhw16i16o = 69; 
      public static final int OIdhw16o16i = 70; 
      public static final int Oidhw16o = 71; 
      public static final int Odhwi16o = 72; 
      public static final int OIdhw8i16o2i = 73; 
      public static final int gOwi8o = 74; 
      public static final int gOIw8o8i = 75; 
      public static final int gOIw8i8o = 76; 
      public static final int gOIw16i16o = 77; 
      public static final int gOIw16o16i = 78; 
      public static final int gOiw16o = 79; 
      public static final int gOwi16o = 80; 
      public static final int gOIw8i16o2i = 81; 
      public static final int gOIw8o16i2o = 82; 
      public static final int gIOw16o16i = 83; 
      public static final int hwigo_s8s8 = 84; 
      public static final int gOIhw8i8o = 85; 
      public static final int gOIhw16i16o = 86; 
      public static final int gOIhw4i16o4i = 87; 
      public static final int gOIhw4i16o4i_s8s8 = 88; 
      public static final int gOIhw8i16o2i = 89; 
      public static final int gOIhw8o16i2o = 90; 
      public static final int gOIhw8o8i = 91; 
      public static final int gOIhw16o16i = 92; 
      public static final int gIOhw16o16i = 93; 
      public static final int gOihw8o = 94; 
      public static final int gOihw16o = 95; 
      public static final int gOhwi8o = 96; 
      public static final int gOhwi16o = 97; 
      public static final int Goihw8g = 98; 
      public static final int Goihw16g = 99; 
      public static final int gOhIw16o4i = 100; 
      public static final int gOIdhw8i8o = 101; 
      public static final int gOIdhw8o8i = 102; 
      public static final int gOdhwi8o = 103; 
      public static final int gOIdhw8i16o2i = 104; 
      public static final int gOIdhw16i16o = 105; 
      public static final int gOIdhw16o16i = 106; 
      public static final int gOidhw16o = 107; 
      public static final int gOdhwi16o = 108; 
      public static final int wino_fmt = 109; 
      public static final int ldigo_p = 110; 
      public static final int ldgoi_p = 111; 
      public static final int format_last = 112; 
    }

    public static class Format2 {
        public static final int format_undef = 0;
        public static final int any = 1;
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
        public static final int OIdhw8i16o2i = 28;
        public static final int OIhw8o16i2o = 29;
        public static final int OIhw8o8i = 30;
        public static final int OIhw16o16i = 31;
        public static final int IOhw16o16i = 32;
        public static final int Oihw8o = 33;
        public static final int Oihw16o = 34;
        public static final int Ohwi8o = 35;
        public static final int Ohwi16o = 36;
        public static final int OhIw16o4i = 37;
        public static final int goihw = 38;
        public static final int hwigo = 39;
        public static final int gOIhw8i8o = 40;
        public static final int gOIhw16i16o = 41;
        public static final int gOIhw4i16o4i = 42;
        public static final int gOIhw8i16o2i = 43;
        public static final int gOIdhw8i16o2i = 44;
        public static final int gOIhw8o16i2o = 45;
        public static final int gOIhw8o8i = 46;
        public static final int gOIhw16o16i = 47;
        public static final int gIOhw16o16i = 48;
        public static final int gOihw8o = 49;
        public static final int gOihw16o = 50;
        public static final int gOhwi8o = 51;
        public static final int gOhwi16o = 52;
        public static final int Goihw8g = 53;
        public static final int Goihw16g = 54;
        public static final int gOhIw16o4i = 55;
        public static final int goidhw = 56;
        public static final int gOIdhw16i16o = 57;
        public static final int gOIdhw16o16i = 58;
        public static final int gOidhw16o = 59;
        public static final int gOdhwi16o = 60;
        public static final int ntc = 61;
        public static final int tnc = 62;
        public static final int ldsnc = 63;
        public static final int ldigo = 64;
        public static final int ldigo_p = 65;
        public static final int ldgoi = 66;
        public static final int ldgoi_p = 67;
        public static final int ldgo = 68;
        public static final int wino_fmt = 69;
        public static final int format_last = 70;
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
    public native static int[] GetPaddingShape(long desc);
    public native static int GetLayout(long desc);
}
