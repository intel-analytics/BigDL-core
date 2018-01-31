package com.intel.analytics.bigdl.mkl;

import java.nio.FloatBuffer;

public class Memory {
    public static class Format {
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
}
