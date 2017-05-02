package com.intel.analytics.bigdl.fixpoint;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

public class FixPoint {
    private static boolean isLoaded = false;
    private static File tmpFile = null;

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

    public native static long FixConvOpCreate(int layout);

    public native static void FixConvOpSetupConvParameter(long desc, long channelOut, long channelIn,
                                                          long group, long kernelHeight, long kernelWidth,
                                                          long strideHeight, long strideWidth, long dilationHeight,
                                                          long dilatioinWidth, long padHeight, long padWidth,
                                                          float[] weight, long weightOffset,
                                                          boolean withBias, float[] bias, long biasOffset,
                                                          boolean relu);

    public native static void FixConvOpQuantizeKernel(long desc, float threshold);

    public native static void FixConvOpQuantizeData(long desc, long batchSize, long channels,
                                                    long inputHeight, long inputWidth, float[] src, long srcOffset,
                                                    float threshold);

    public native static void FixConvOpSetupTargetBuffer(long desc, float[] dst, long dstOffset);

    public native static void FixConvOpExecute(long desc, float fault_tolerance);

    public native static void FixConvOpFree(long desc);
}
