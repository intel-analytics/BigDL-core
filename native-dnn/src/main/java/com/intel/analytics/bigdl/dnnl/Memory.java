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

package com.intel.analytics.bigdl.dnnl;

public class Memory {
    static {
        DNNL.isLoaded();
    }

    public static class FormatKind {
        public static final int Undef = 0;
        public static final int Any = 1;
        public static final int Blocked = 2;
        public static final int FormatKindWino = 3;
        public static final int FormatKindRnnPacked = 4;
    }

    public static class ExtraFlags {
        public static final int None = 0;
        public static final int CompensationConvS8s8 = 1;
        public static final int ScaleAdjust = 2;
        public static final int GpuRnnU8s8Compensation = 4;
    }

    public static class FormatTag {
        public static final int undef = 0;
        public static final int any = 1;
        public static final int a = 2;
        public static final int ab = 3;
        public static final int abc = 4;
        public static final int abcd = 5;
        public static final int abcde = 6;
        public static final int abcdef = 7;
        public static final int abdec = 8;
        public static final int acb = 9;
        public static final int acbde = 10;
        public static final int acdb = 11;
        public static final int acdeb = 12;
        public static final int ba = 13;
        public static final int bac = 14;
        public static final int bacd = 15;
        public static final int bca = 16;
        public static final int bcda = 17;
        public static final int bcdea = 18;
        public static final int cba = 19;
        public static final int cdba = 20;
        public static final int cdeba = 21;
        public static final int decab = 22;
        public static final int Abc16a = 23;
        public static final int ABc16a16b = 24;
        public static final int aBc16b = 25;
        public static final int ABc16b16a = 26;
        public static final int Abc4a = 27;
        public static final int aBc4b = 28;
        public static final int ABc4b16a4b = 29;
        public static final int ABc4b4a = 30;
        public static final int ABc8a16b2a = 31;
        public static final int ABc8a8b = 32;
        public static final int aBc8b = 33;
        public static final int ABc8b16a2b = 34;
        public static final int BAc8a16b2a = 35;
        public static final int ABc8b8a = 36;
        public static final int Abcd16a = 37;
        public static final int ABcd16a16b = 38;
        public static final int ABcd32a32b = 39;
        public static final int aBcd16b = 40;
        public static final int ABcd16b16a = 41;
        public static final int aBCd16b16c = 42;
        public static final int aBCd16c16b = 43;
        public static final int Abcd4a = 44;
        public static final int aBcd4b = 45;
        public static final int ABcd4b16a4b = 46;
        public static final int ABcd4b4a = 47;
        public static final int aBCd4c16b4c = 48;
        public static final int aBCd4c4b = 49;
        public static final int ABcd8a16b2a = 50;
        public static final int ABcd8a8b = 51;
        public static final int aBcd8b = 52;
        public static final int ABcd8b16a2b = 53;
        public static final int aBCd8b16c2b = 54;
        public static final int BAcd8a16b2a = 55;
        public static final int ABcd8b8a = 56;
        public static final int aBCd8b8c = 57;
        public static final int aBCd8c16b2c = 58;
        public static final int ABcde8a16b2a = 59;
        public static final int aCBd8b16c2b = 60;
        public static final int aBCd8c8b = 61;
        public static final int Abcde16a = 62;
        public static final int ABcde16a16b = 63;
        public static final int BAcde8a16b2a = 64;
        public static final int aBcde16b = 65;
        public static final int ABcde16b16a = 66;
        public static final int aBCde16b16c = 67;
        public static final int aBCde16c16b = 68;
        public static final int aBCde2c8b4c = 69;
        public static final int Abcde4a = 70;
        public static final int aBcde4b = 71;
        public static final int ABcde4b4a = 72;
        public static final int aBCde4b4c = 73;
        public static final int aBCde4c16b4c = 74;
        public static final int aBCde4c4b = 75;
        public static final int Abcde8a = 76;
        public static final int ABcde8a8b = 77;
        public static final int BAcde16b16a = 78;
        public static final int aBcde8b = 79;
        public static final int ABcde8b16a2b = 80;
        public static final int aBCde8b16c2b = 81;
        public static final int aCBde8b16c2b = 82;
        public static final int ABcde8b8a = 83;
        public static final int aBCde8b8c = 84;
        public static final int ABcd4a8b8a4b = 85;
        public static final int ABcd2a8b8a2b = 86;
        public static final int aBCde4b8c8b4c = 87;
        public static final int aBCde2b8c8b2c = 88;
        public static final int aBCde8c16b2c = 89;
        public static final int aBCde8c8b = 90;
        public static final int aBcdef16b = 91;
        public static final int aBCdef16b16c = 92;
        public static final int aBCdef16c16b = 93;
        public static final int aBcdef4b = 94;
        public static final int aBCdef4c4b = 95;
        public static final int aBCdef8b8c = 96;
        public static final int aBCdef8c16b2c = 97;
        public static final int aBCdef8b16c2b = 98;
        public static final int aCBdef8b16c2b = 99;
        public static final int aBCdef8c8b = 100;
        public static final int aBdc16b = 101;
        public static final int aBdc4b = 102;
        public static final int aBdc8b = 103;
        public static final int aBdec16b = 104;
        public static final int aBdec32b = 105;
        public static final int aBdec4b = 106;
        public static final int aBdec8b = 107;
        public static final int aBdefc16b = 108;
        public static final int aCBdef16c16b = 109;
        public static final int aBdefc4b = 110;
        public static final int aBdefc8b = 111;
        public static final int Abcdef16a = 112;
        public static final int Acb16a = 113;
        public static final int Acb4a = 114;
        public static final int Acb8a = 115;
        public static final int aCBd16b16c = 116;
        public static final int aCBd16c16b = 117;
        public static final int aCBde16b16c = 118;
        public static final int aCBde16c16b = 119;
        public static final int Acdb16a = 120;
        public static final int Acdb32a = 121;
        public static final int Acdb4a = 122;
        public static final int Acdb8a = 123;
        public static final int Acdeb16a = 124;
        public static final int Acdeb4a = 125;
        public static final int Acdeb8a = 126;
        public static final int BAc16a16b = 127;
        public static final int BAc16b16a = 128;
        public static final int BAcd16a16b = 129;
        public static final int BAcd16b16a = 130;
        public static final int format_tag_last = 131;
        public static final int x = 2;
        public static final int nc = 3;
        public static final int cn = 13;
        public static final int tn = 3;
        public static final int nt = 13;
        public static final int ncw = 4;
        public static final int nwc = 9;
        public static final int nchw = 5;
        public static final int nhwc = 11;
        public static final int chwn = 17;
        public static final int ncdhw = 6;
        public static final int ndhwc = 12;
        public static final int oi = 3;
        public static final int io = 13;
        public static final int oiw = 4;
        public static final int owi = 9;
        public static final int wio = 19;
        public static final int iwo = 16;
        public static final int oihw = 5;
        public static final int hwio = 20;
        public static final int ohwi = 11;
        public static final int ihwo = 17;
        public static final int iohw = 15;
        public static final int oidhw = 6;
        public static final int dhwio = 21;
        public static final int odhwi = 12;
        public static final int idhwo = 18;
        public static final int goiw = 5;
        public static final int goihw = 6;
        public static final int hwigo = 22;
        public static final int giohw = 10;
        public static final int goidhw = 7;
        public static final int tnc = 4;
        public static final int ntc = 14;
        public static final int ldnc = 5;
        public static final int ldigo = 6;
        public static final int ldgoi = 8;
        public static final int ldgo = 5;
        public static final int nCdhw16c = 65;
        public static final int nCdhw4c = 71;
        public static final int nCdhw8c = 79;
        public static final int nChw16c = 40;
        public static final int nChw4c = 45;
        public static final int nChw8c = 52;
        public static final int nCw16c = 25;
        public static final int nCw4c = 28;
        public static final int nCw8c = 33;
        public static final int NCw16n16c = 24;
        public static final int NCdhw16n16c = 63;
        public static final int NChw16n16c = 38;
        public static final int NChw32n32c = 39;
        public static final int IOw16o16i = 127;
        public static final int IOw16i16o = 128;
        public static final int OIw16i16o = 26;
        public static final int OIw16o16i = 24;
        public static final int Oiw16o = 23;
        public static final int OIw4i16o4i = 29;
        public static final int OIw4i4o = 30;
        public static final int Oiw4o = 27;
        public static final int OIw8i16o2i = 34;
        public static final int OIw8i8o = 36;
        public static final int OIw8o16i2o = 31;
        public static final int IOw8o16i2o = 35;
        public static final int OIw8o8i = 32;
        public static final int Owi16o = 113;
        public static final int Owi4o = 114;
        public static final int Owi8o = 115;
        public static final int IOhw16i16o = 130;
        public static final int IOhw16o16i = 129;
        public static final int Ohwi16o = 120;
        public static final int Ohwi32o = 121;
        public static final int Ohwi4o = 122;
        public static final int Ohwi8o = 123;
        public static final int OIhw16i16o = 41;
        public static final int OIhw16o16i = 38;
        public static final int Oihw16o = 37;
        public static final int OIhw4i16o4i = 46;
        public static final int OIhw4i4o = 47;
        public static final int Oihw4o = 44;
        public static final int OIhw8i16o2i = 53;
        public static final int OIhw8i8o = 56;
        public static final int OIhw8o16i2o = 50;
        public static final int IOhw8o16i2o = 55;
        public static final int OIhw8o8i = 51;
        public static final int Odhwi16o = 124;
        public static final int Odhwi4o = 125;
        public static final int Odhwi8o = 126;
        public static final int OIdhw16i16o = 66;
        public static final int OIdhw16o16i = 63;
        public static final int Oidhw16o = 62;
        public static final int OIdhw4i4o = 72;
        public static final int Oidhw4o = 70;
        public static final int OIdhw8i16o2i = 80;
        public static final int OIdhw8i8o = 83;
        public static final int OIdhw8o16i2o = 59;
        public static final int IOdhw8o16i2o = 64;
        public static final int OIdhw8o8i = 77;
        public static final int IOdhw16i16o = 78;
        public static final int Goiw16g = 37;
        public static final int gIOw16o16i = 116;
        public static final int gIOw16i16o = 117;
        public static final int gOIw16i16o = 43;
        public static final int gOIw16o16i = 42;
        public static final int gOiw16o = 40;
        public static final int gOIw4i16o4i = 48;
        public static final int gOIw4i4o = 49;
        public static final int gOiw4o = 45;
        public static final int gOIw8i16o2i = 58;
        public static final int gOIw8i8o = 61;
        public static final int gOIw8o16i2o = 54;
        public static final int gIOw8o16i2o = 60;
        public static final int gOIw8o8i = 57;
        public static final int gOwi16o = 101;
        public static final int gOwi4o = 102;
        public static final int gOwi8o = 103;
        public static final int gIOhw16i16o = 119;
        public static final int gIOhw16o16i = 118;
        public static final int gOhwi16o = 104;
        public static final int gOhwi32o = 105;
        public static final int gOhwi4o = 106;
        public static final int gOhwi8o = 107;
        public static final int Goihw16g = 62;
        public static final int gOIhw16i16o = 68;
        public static final int gOIhw16o16i = 67;
        public static final int gOihw16o = 65;
        public static final int gOIhw2i8o4i = 69;
        public static final int gOIhw4i16o4i = 74;
        public static final int gOIhw4i4o = 75;
        public static final int gOIhw4o4i = 73;
        public static final int gOihw4o = 71;
        public static final int Goihw8g = 76;
        public static final int gOIhw8i16o2i = 89;
        public static final int gOIhw8i8o = 90;
        public static final int gOIhw8o16i2o = 81;
        public static final int gIOhw8o16i2o = 82;
        public static final int gOIhw8o8i = 84;
        public static final int OIhw4o8i8o4i = 85;
        public static final int OIhw2o8i8o2i = 86;
        public static final int gOIhw4o8i8o4i = 87;
        public static final int gOIhw2o8i8o2i = 88;
        public static final int gIOdhw16i16o = 109;
        public static final int gOdhwi16o = 108;
        public static final int gOdhwi4o = 110;
        public static final int gOdhwi8o = 111;
        public static final int gOIdhw16i16o = 93;
        public static final int gOIdhw16o16i = 92;
        public static final int gOidhw16o = 91;
        public static final int gOIdhw4i4o = 95;
        public static final int gOidhw4o = 94;
        public static final int gOIdhw8i16o2i = 97;
        public static final int gOIdhw8i8o = 100;
        public static final int gOIdhw8o16i2o = 98;
        public static final int gIOdhw8o16i2o = 99;
        public static final int gOIdhw8o8i = 96;
        public static final int Goidhw16g = 112;
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

    public native static long CopyPtr2ByteArray(long src, int srcOffset, byte[] dst, int dstOffset,
                                                int length, int elementSize);

    public native static long CopyPtr2IntArray(long src, int srcOffset, int[] dst, int dstOffset,
                                                int length, int elementSize);

    public native static long AlignedMalloc(int capacity, int size);
    public native static void AlignedFree(long ptr);

    public native static void SAdd(int n, long aPtr, int aOffset, long bPtr, int bOffset,
                                    long yPtr, int yOffset);

    public native static void Scale(int n, float scaleFactor, long from, long to);
    public native static void Axpby(int n, float a, long x, float b, long y);
    public native static void Set(long data, float value, int length, int elementSize);

    public native static long[] GetShape(long desc);
    public native static long[] GetPaddingShape(long desc);
    public native static int GetLayout(long desc);
    public native static int GetDataType(long desc);
    public native static long GetSize(long desc);
}
