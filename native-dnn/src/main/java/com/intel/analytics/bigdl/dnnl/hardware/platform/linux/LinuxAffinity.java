package com.intel.analytics.bigdl.dnnl.hardware.platform.linux;

import com.intel.analytics.bigdl.dnnl.hardware.platform.IAAffinity;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class LinuxAffinity implements IAAffinity {
    public static native int setAffinity0(int[] set);
    public static native int getAffinity0(int[] set);
    public static native int[] setOmpAffinity0(int[] set);
    public static native int[] getOmpAffinity0(int coreCounts);

    private int coreCounts;
    private int[] coreList;
    private ConcurrentHashMap<Integer, List<Long>> availableCores;

    public LinuxAffinity(int coreCounts) {
        this.coreCounts = coreCounts;

        this.coreList = getAffinity();

        this.availableCores = new ConcurrentHashMap<Integer, List<Long>>();
        for (int key : this.coreList) {
            availableCores.put(key, new ArrayList<Long>());
        }
    }

    public synchronized void setAffinity() {
        int min = Integer.MAX_VALUE;
        int targetCore = 0; // by default, use the first core of coreList.
        for (int key : this.coreList) {
            int size = availableCores.get(key).size();
            if (size < min) {
                min = size;
                targetCore = key;
            }
        }

        int[] set = { targetCore };
        bindToCores(set);

        for (int key : set) {
            availableCores.get(key).add(Thread.currentThread().getId());
        }
    }

    public synchronized void setAffinity(int[] set) {
        boolean isAllAvailable = true;
        for (int key : set) {
            if (!availableCores.containsKey(key)) {
                isAllAvailable = false;
            }
        }

        if (!isAllAvailable) {
            throw new BindException(set);
        }

        bindToCores(set);

        for (int key : set) {
            availableCores.get(key).add(Thread.currentThread().getId());
        }
    }

    public synchronized void resetAffinity() {
        int[] affinity = this.getAffinity();
        for (int i : affinity) {
            if (!availableCores.containsKey(i)) {
                throw new RuntimeException("Thread " +
                        Thread.currentThread().getName() + " #id " +
                        Thread.currentThread().getId() +
                        " has bind to wrong cores");
            }
        }

        bindToCores(this.coreList); // reset to default

        for (int key : affinity) {
            availableCores.get(key).remove(Thread.currentThread().getId());
        }
    }

    public synchronized void setOmpAffinity() {
        // TODO omp thread can be managed availableCores.
        int[] ret = setOmpAffinity0(coreList);

        boolean isAllSuccess = true;

        for (int r : ret) {
            if (r != 0) {
                isAllSuccess = false;
                break;
            }
        }

        if (!isAllSuccess) {
            throw new BindException(coreList);
        }
    }

    public synchronized int[] getAffinity() {
        int[] temp = new int[coreCounts];
        Arrays.fill(temp, 0);

        if (getAffinity0(temp) != 0) {
            throw new RuntimeException("failed to get the affinity info of thread " +
                    Thread.currentThread().getName() +
                    " #id " + Thread.currentThread().getId());
        }

        return humanReadable(temp);
    }

    public synchronized int[] getOmpAffinity() {
        int[] ret = getOmpAffinity0(this.coreCounts);

        boolean isAllSucc = true;

        for (int i = 0; i < ret.length; i++) {
            if (ret[i] == -1) {
                isAllSucc = false;
                break;
            }
        }

        if (!isAllSucc) {
            throw new RuntimeException("failed to get the affinity info of thread " +
                    Thread.currentThread().getName() +
                    " #id " + Thread.currentThread().getId());
        }

        return ret;
    }

    public Map<Integer, List<Long>> stats() {
        return Collections.unmodifiableMap(availableCores);
    }

    private int[] humanReadable(int[] set) {
        int counts = 0;
        List<Integer> temp = new ArrayList<Integer>();

        for (int i = 0; i < set.length; i++) {
            if (set[i] == 1) {
                temp.add(i);
            }
        }

        int[] result = new int[temp.size()];
        for (int i = 0; i < temp.size(); i++) {
            result[i] = temp.get(i);
        }

        return result;
    }

    private void bindToCores(int[] set) {
        if (setAffinity0(set) != 0) {
            throw new BindException(set);
        }
    }

    class BindException extends RuntimeException {
        StringBuilder message;

        BindException(int[] cpuSet) {
            message = new StringBuilder();
            message.append("fail to bind thread ");
            message.append(Thread.currentThread().getName()).append(" ");
            message.append("#id ").append(Thread.currentThread().getId()).append(" ");

            for (int i = 0; i < cpuSet.length - 1; i++) {
                message.append(cpuSet[i]).append(",");
            }

            message.append(cpuSet[cpuSet.length - 1]);
        }

        @Override
        public String getMessage() {
            return message.toString();
        }
    }
}
