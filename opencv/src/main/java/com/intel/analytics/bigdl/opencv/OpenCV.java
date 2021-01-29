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

package com.intel.analytics.bigdl.opencv;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

/**
 * OpenCV Library Wrapper for JVM
 */
public class OpenCV {
    private static boolean isLoaded = false;
    private static File tmpFile = null;

    static {
        try {
            String jopencvFileName = "libopencv_java320.so";
            if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                jopencvFileName = "libopencv_java320.dylib";
            } else if (System.getProperty("os.name").toLowerCase().contains("win")) {
                jopencvFileName = "opencv_java320.dll";
            }
            // TODO for windows, we don't create mkl.native dir
            Path tempDir = null;
            if (os.contains("win")) {
                tempDir = Paths.get(System.getProperty("java.io.tmpdir"));
            } else {
                tempDir = Files.createTempDirectory("opencv.native.");
            }

            tmpFile = extract(tempDir, jopencvFileName);
            System.load(tmpFile.getAbsolutePath());
            isLoaded = true;
            deleteAll(tempDir);
        } catch (Exception e) {
            isLoaded = false;
            e.printStackTrace();
            throw new RuntimeException("Failed to load OpenCV");
        }
    }

    /**
     * Check if opencv is loaded
     * @return
     */
    public static boolean isOpenCVLoaded() {
        return isLoaded;
    }

    /**
     * Get the temp path of the .so file
     * @return
     */
    public static String getTmpSoFilePath() {
        if (tmpFile == null)
            return "";
        else
            return tmpFile.getAbsolutePath();
    }

    // Extract so file from jar to a temp path
    private static File extract(Path tempDir, String path) {
        try {
            URL url = OpenCV.class.getResource("/" + path);
            if (url == null) {
                throw new Error("Can't find dynamic lib file in jar, path = " + path);
            }

            InputStream in = OpenCV.class.getResourceAsStream("/" + path);
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

    private static void deleteAll(Path tempDir) {
        File dir = tempDir.toFile();
        for (File f: dir.listFiles()) {
            f.delete();
        }

        dir.delete();
    }
}
