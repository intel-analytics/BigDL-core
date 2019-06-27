package com.intel.analytics.bigdl.mkl.hardware.platform.linux;

import com.intel.analytics.bigdl.mkl.hardware.platform.IAAffinity;

import java.util.Arrays;
import java.util.BitSet;

public class LinuxAffinity implements IAAffinity {
    public static native int setAffinity0(int[] sets);
    public static native int getAffinity0(int[] sets, int cpuCounts);
    public static native void setOmpAffinity0(int[] sets);

    private int cpuCounts;
    private boolean[] availableCPUs;

    public LinuxAffinity(int cpuCounts) {
        this.cpuCounts = cpuCounts;
        this.availableCPUs = new boolean[cpuCounts];
        Arrays.fill(this.availableCPUs, false);
    }

    public synchronized int setAffinity() {
        int ret = -1;
        for (int i = 0; i < availableCPUs.length; i++) {
            if (!availableCPUs[i]) {
                ret = i;
                int[] set = {i};
                setAffinity0(set);
                availableCPUs[i] = true;
                break;
            }
        }

        return ret;
    }

    public synchronized void setAffinity(int[] sets) {
        for (int i = 0; i < sets.length; i++) {
            if (sets[i] < cpuCounts) {
                availableCPUs[i] = true;
            }
        }

        setAffinity0(sets);
    }

    public synchronized void setOmpAffinity() {
        setOmpAffinity0(getAffinity());
    }

    public synchronized int[] getAffinity() {
        int[] temp = new int[cpuCounts];
        Arrays.fill(temp, 0);
        getAffinity0(temp, cpuCounts);

        int rIndex = 0;
        int[] ret = new int[getSetted(temp)];

        for (int tIndex = 0; tIndex < cpuCounts; tIndex++) {
            int elem = temp[tIndex];

            if (elem == 1) {
                ret[rIndex] = tIndex;
                rIndex++;
            }
        }

        for (int elem : ret) {
            availableCPUs[elem] = false;
        }

        return ret;
    }

    private int getSetted(int[] sets) {
        int counts  = 0;
        for (int set : sets) {
            if (set == 1) {
                counts++;
            }
        }

        return counts;
    }
}
