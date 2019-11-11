package com.intel.analytics.bigdl.dnnl.hardware.platform.linux;

import com.intel.analytics.bigdl.dnnl.hardware.platform.IACpuInfo;

import java.io.*;
import java.util.HashSet;
import java.util.Set;

public class LinuxCpuInfo implements IACpuInfo {
    private final int sockets;
    private final int physicalCores;
    private final int logicalCores;

    public LinuxCpuInfo() {
        String defaultPath = "/proc/cpuinfo";
        try {
            int processors = parseCpuInfo("processor", defaultPath).size();
            int siblings = getSiblings(parseCpuInfo("siblings", defaultPath));
            int coreIds = parseCpuInfo("core id", defaultPath).size();
            int physicalIds = parseCpuInfo("physical id", defaultPath).size();
            this.logicalCores = processors;
            this.sockets = physicalIds;

            if (siblings == coreIds) {
                this.physicalCores = processors;
            } else {
                this.physicalCores = coreIds * physicalIds;
            }
        } catch (IOException e) {
            throw new AssertionError();
        }

    }

    private int getSiblings(Set siblings) {
        String value = siblings.toArray()[0].toString();
        return Integer.parseInt(value);
    }

    private Set parseCpuInfo(String pattern, String path) throws IOException {
        String line;
        Set set = new HashSet();
        FileInputStream inputStream = new FileInputStream(path);
        InputStreamReader inputReader = new InputStreamReader(inputStream, "UTF-8");
        BufferedReader bufferedReader = new BufferedReader(inputReader);

        while ((line = bufferedReader.readLine()) != null) {
            if (line.trim().length() == 0) {
                continue;
            }

            String[] words = line.trim().split("\\s*:\\s*", 2);
            if (words[0].equals(pattern) && words.length > 1) {
                set.add(words[1]);
            }
        }

        return set;
    }

    public int getPhysicalProcessorCount() {
        return this.physicalCores;
    }

    public int getLogicalProcessorCount() {
        return this.logicalCores;
    }

    public int getSocketsCount() {
       return this.sockets;
    }

    public boolean isEnableHyperThreading() {
        boolean ret = true;

        ret &= logicalCores % physicalCores == 0;
        ret &= logicalCores / physicalCores == 2;

        return ret;
    }
}
