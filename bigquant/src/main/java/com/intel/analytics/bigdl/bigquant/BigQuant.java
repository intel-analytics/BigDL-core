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

package com.intel.analytics.bigdl.bigquant;

public class BigQuant {
    private static boolean isLoaded = false;
    public final static int NCHW = 0;
    public final static int NHWC = 1;

    static {
        try {
            Loader loader = new Loader();
            loader.init();

            isLoaded = true;
        } catch (Exception e) {
            isLoaded = false;

            e.printStackTrace();
            throw new RuntimeException("Failed to load Quant");
        }
    }

    public static void main(String[] args) {
        printHello();
    }

    public native static void printHello();

    public native static int loadRuntime(String path);

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

    public native static void MixPrecisionGEMM(int layout,
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
