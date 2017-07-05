package com.intel.analytics.bigdl.fixpoint;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

public class FixPoint {
    private static boolean isLoaded = false;
    private static File tmpFile = null;

    public final static int NCHW = 0;
    public final static int NHWC = 1;

    static {
        try {
            String fixpointName = "libfixpoint.so";
            if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                fixpointName = "libfixpoint.dylib";
            }
            tmpFile = extract(fixpointName);
            System.load(tmpFile.getAbsolutePath());
            tmpFile.delete(); // delete so temp file after loaded
            isLoaded = true;

        } catch (Exception e) {
            isLoaded = false;
            e.printStackTrace();
            // TODO: Add an argument for user, continuing to run even if MKL load failed.
            throw new RuntimeException("Failed to load FixPoint");
        }
    }

    // Extract so file from jar to a temp path
    private static File extract(String path) {
        try {
            URL url = FixPoint.class.getResource("/" + path);
            if (url == null) {
                throw new Error("Can't find so file in jar, path = " + path);
            }

            InputStream in = FixPoint.class.getResourceAsStream("/" + path);
            File file = createTempFile("dlNativeLoader", path);

            ReadableByteChannel src = newChannel(in);
            FileChannel dest = new FileOutputStream(file).getChannel();
            dest.transferFrom(src, 0, Long.MAX_VALUE);
            return file;
        } catch (Throwable e) {
            throw new Error("Can't extract so file to /tmp dir");
        }
    }

    public native static void printHello();


    public native static long FixConvKernelDescInit(int c_out,
                                                    int c_in,
                                                    int kernel_h,
                                                    int kernel_w);

     public native static void FixConvKernelInit(long fix_tensor,
                                                 float[] src,
                                                 int srcOffset,
                                                 int c_out,
                                                 int c_in,
                                                 int kernel_h,
                                                 int kernel_w,
                                                 float threshold,
                                                 int layout);

     public native static void FixConvKernelLoadFromModel(char[] src,
                                                          int srcOffset,
                                                          float[] min,
                                                          float[] max,
                                                          int c_out,
                                                          int c_in,
                                                          int kernel_h,
                                                          int kernel_w,
                                                          float threshold,
                                                          int layout);

     public native static long FixConvDataDescInit(int c_in,
                                                   int kernel_h,
                                                   int kernel_w,
                                                   int stride_h,
                                                   int stride_w,
                                                   int pad_h,
                                                   int pad_w,
                                                   int dilation_h,
                                                   int dilation_w,
                                                   int batch_size,
                                                   int h_in,
                                                   int w_in);

    public native static void FixConvDataInit(long fix_tensor,
                                              float[] src, int srcOffset,
                                              int c_in,
                                              int kernel_h,
                                              int kernel_w,
                                              int stride_h,
                                              int stride_w,
                                              int pad_h,
                                              int pad_w,
                                              int dilation_h,
                                              int dilation_w,
                                              int batch_size,
                                              int h_in,
                                              int w_in,
                                              float threshold,
                                              int layout);

    public native static long FixConvKernelSumDescInit(int c_out);

    public native static void FixConvKernelSumInit(long fp_tensor,
                                                   float[] src, int srcOffset,
                                                   int n,
                                                   int c,
                                                   int h,
                                                   int w);

    public native static void InternalMixPrecisionConvolutionGEMM(int layout,
                                                                  long pa,
                                                                  long pb,
                                                                  float[] pc, int pcOffset,
                                                                  int m,
                                                                  int n,
                                                                  int k,
                                                                  long kernelSum,
                                                                  float[] bias,
                                                                  int biasOffset,
                                                                  int batch_size,
                                                                  int channel_per_group,
                                                                  int height_out,
                                                                  int width_out,
                                                                  float fault_tolerance);

}
