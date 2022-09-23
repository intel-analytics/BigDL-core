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

package com.intel.analytics.bigdl.ckks;

import java.io.*;
import java.lang.Exception;
import java.nio.ByteBuffer;
import java.util.List;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.util.List;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

/**
 * MKL Library Wrapper for JVM
 */
public class CKKS {
    private static final boolean DEBUG =
            System.getProperty("com.intel.analytics.bigdl.ckks.CKKS.DEBUG") != null;

    private static boolean isLoaded = false;
    private static File tmpFile = null;
    private static String os = System.getProperty("os.name").toLowerCase();

    static {
        String[] LIBS = new String[]{
          "libvflhe_jni.so"};

        isLoaded = tryLoadLibrary(LIBS);
        if (!isLoaded) {
            try {
                Path tempDir = Files.createTempDirectory("bigdl.ckks.");
                for (int i = 0; i < LIBS.length; i++) {
                    String libName = LIBS[i];
                    log("[DEBUG] Loading " + libName);
                    if (CKKS.class.getResource("/" + libName) != null) {
                        try {
                            tmpFile = extract(tempDir, libName);
                            System.load(tmpFile.getAbsolutePath());
                        } catch (Exception e) {
                            throw new UnsatisfiedLinkError(
                                    String.format(
                                            "Unable to extract & load (%s)", e.toString()));
                        }
                        log("[DEBUG] Loaded " + libName);
                    }
                }
                isLoaded = true;
                deleteAll(tempDir);
                log("[DEBUG] delete tempdir");
            } catch (Exception e) {
                isLoaded = false;
                e.printStackTrace();
                // TODO: Add an argument for user, continuing to run even if MKL load failed.
                throw new RuntimeException("Failed to load MKL");
            }
        }
    }

    // Extract so file from jar to a temp path
    private static File extract(Path tempDir, String path) {
        try {
            URL url = CKKS.class.getResource("/" + path);
            if (url == null) {
                throw new Error("Can't find dynamic lib file in jar, path = " + path);
            }

            InputStream in = CKKS.class.getResourceAsStream("/" + path);
            File file = null;

            file = new File(tempDir.toFile() + File.separator + path);

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

    private static void log(String msg) {
        if (DEBUG) {
            System.err.println("com.intel.analytics.bigdl.mkl: " + msg);
        }
    }

    private static void deleteAll(Path tempDir) {
        File dir = tempDir.toFile();
        for (File f: dir.listFiles()) {
            f.delete();
        }

        dir.delete();
    }

    private static boolean tryLoadLibrary(String[] libs) {
        log("try loading native libraries from java.library.path");
        try {
            for (int i = 0; i < libs.length; i++) {
                String libName = libs[i];
                if (libName.indexOf(".") != -1) {
                    // Remove lib and .so
                    libName = libName.substring(3, libName.indexOf("."));
                }
                System.loadLibrary(libName);
                log("[DEBUG] loaded " + libName + " from java.library.path");
            }
            log("[DEBUG] Loaded native libraries from java.library.path");
            return true;
        } catch (UnsatisfiedLinkError e) {
            log("tryLoadLibraryFailed: " + e.getMessage());
            return false;
        }

    }

    public static void saveSecret(byte[][] secret, String path) throws IOException {
        FileOutputStream fos = new FileOutputStream(path);
        fos.write(ByteBuffer.allocate(4).putInt(secret.length).array());
        for (byte[] bytes : secret) {
          fos.write(ByteBuffer.allocate(4).putInt(bytes.length).array());
          fos.write(bytes);
        }
        fos.close();
    }

    public static byte[][] loadSecret(String path) throws IOException{
      FileInputStream fis = new FileInputStream(path);
      byte[] buffer = new byte[4];
      fis.read(buffer);
      int numSecret = ByteBuffer.wrap(buffer).getInt();
      byte[][] secret = new byte[numSecret][];
      for (int i = 0; i < numSecret; i++) {
          fis.read(buffer);
          int len = ByteBuffer.wrap(buffer).getInt();
          byte[] bytes = new byte[len];
          fis.read(bytes);
          secret[i] = bytes;
      }
      return secret;
    }

    public native byte[][] createSecrets();

    public native long createCkksEncryptor(byte[][] secret);

    public native byte[] ckksEncrypt(long encryptor, float[] data);

    public native float[] ckksDecrypt(long encryptor, byte[] data);

    public native long createCkksCommonInstance(byte[][] secret);

    public native byte[][] train(long ckksCommon, byte[] output, byte[] target);

    public native byte[][] backward(long ckksCommon, byte[] output, byte[] target);

    public native byte[][] sigmoidForward(long ckksCommon, byte[] output);

    public native byte[] cadd(long ckksCommon, byte[] input1, byte[] input2);
}
