package com.intel.analytics.bigdl.bigquant;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

public class BigQuant {
    private static boolean isLoaded = false;
    private static File tmpFile = null;

    public final static int NCHW = 0;
    public final static int NHWC = 1;

    private static String getLibraryName(String name) {
        String os = System.getProperty("os.name").toLowerCase();
        String suffix = ".so";

        if (os.contains("mac")) {
            suffix = ".dylib";
        } else if (os.contains("win")) {
            suffix = ".dll";
        }

        name = "lib" + name + suffix;

        return name;
    }

    private static void loadLibary(String name) {
        tmpFile = extract(getLibraryName(name));
        try {
            System.load(tmpFile.getAbsolutePath());
        } finally {
            tmpFile.delete(); // delete so temp file after loaded
        }
    }

    static {
        try {
            String resourceDir = BigQuant.class.getProtectionDomain().getCodeSource().getLocation().getPath();
            loadLibary("bigquant_rt");
            loadLibary("bigquant");
            loadRuntime(resourceDir);
            isLoaded = true;

        } catch (Exception e) {
            isLoaded = false;
            e.printStackTrace();
            throw new RuntimeException("Failed to load Quant");
        }
    }

    // Extract so file from jar to a temp path
    private static File extract(String path) {
        try {
            URL url = BigQuant.class.getResource("/" + path);
            if (url == null) {
                throw new Error("Can't find dynamic lib file in jar, path = " + path);
            }

            InputStream in = BigQuant.class.getResourceAsStream("/" + path);
            File file = null;

            // Windows won't allow to change the dll name, so we keep the name
            // It's fine as windows is consider in a desktop env, so there won't multiple instance
            // produce the dynamic lib file
            if (System.getProperty("os.name").toLowerCase().contains("win")) {
                file = new File(System.getProperty("java.io.tmpdir") + File.separator + path);
            } else {
                file = createTempFile("dlNativeLoader", path);
            }

            ReadableByteChannel src = newChannel(in);
            FileChannel dest = new FileOutputStream(file).getChannel();
            dest.transferFrom(src, 0, Long.MAX_VALUE);
            dest.close();
            src.close();
            return file;
        } catch (Throwable e) {
            throw new Error("Can't extract dynamic lib file to /tmp dir.\n" + e);
        }
    }

    public static void main(String[] args) {
        printHello();
    }

    public native static void printHello();

    public native static void loadRuntime(String path);

    public native static long ConvKernelDescInit(int c_out,
                                                 int c_in,
                                                 int kernel_h,
                                                 int kernel_w);

     public native static void ConvKernelInit(long tensor,
                                              float[] src,
                                              int srcOffset,
                                              int c_out,
                                              int c_in,
                                              int kernel_h,
                                              int kernel_w,
                                              float threshold,
                                              int layout);

     public native static void ConvKernelLoadFromModel(long tensor,
                                                       byte[] src,
                                                       int srcOffset,
                                                       float[] min,
                                                       float[] max,
                                                       int c_out,
                                                       int c_in,
                                                       int kernel_h,
                                                       int kernel_w,
                                                       float threshold,
                                                       int layout);

     public native static long ConvDataDescInit(int c_in,
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

    public native static void ConvDataInit(long tensor,
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

    public native static long ConvKernelSumDescInit(int c_out);

    public native static void ConvKernelSumInit(long tensor,
                                                float[] src, int srcOffset,
                                                int n,
                                                int c,
                                                int h,
                                                int w);

    public native static void InternalMixPrecisionConvolutionGEMM(int layout,
                                                                  long pa,
                                                                  long pb,
                                                                  float[] pc, int pcOffset,
                                                                  float[] kernelSum, int kernelSumOffset,
                                                                  float[] bias, int biasOffset,
                                                                  int batch_size,
                                                                  int channel_per_group,
                                                                  int height_out,
                                                                  int width_out,
                                                                  float fault_tolerance);
    public native static void FreeMemory(long ptr);

    public native static long FCKernelDescInit(int c_out, int c_in);

    public native static void FCKernelLoadFromModel(long tensor,
                                                    byte[] src,
                                                    float[] min,
                                                    float[] max,
                                                    int c_out,
                                                    int c_in,
                                                    float threshold,
                                                    int layout);

    public native static long FCDataDescInit(int batch_size,
                                             int channel);

    public native static void FCDataInit(long tensor,
                                         float[] src, int srcOffset,
                                         int batch_size,
                                         int channel,
                                         float threshold,
                                         int layout);
}
