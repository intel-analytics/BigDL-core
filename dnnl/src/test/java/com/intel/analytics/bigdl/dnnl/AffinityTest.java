package com.intel.analytics.bigdl.dnnl;

import com.intel.analytics.bigdl.dnnl.hardware.Affinity;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertTrue;

public class AffinityTest {
    @Test
    public void GetAffinity() {
        int[] affinity = Affinity.getAffinity();
        Set<Integer> set = Affinity.stats().keySet();

        Arrays.sort(affinity);

        assertTrue(affinity.length == set.size());

        for (int key : affinity) {
            assertTrue(set.contains(key));
        }
    }

    @Test
    public void SetSingleAffinity() {
        int[] backup = Affinity.getAffinity();

        Affinity.setAffinity();

        int[] cores = Affinity.getAffinity();

        assertTrue(cores.length == 1);
        assertTrue(cores[0] == backup[0]);

        Affinity.resetAffinity();
    }

    @Test
    public void SetMultiAffinity() {
        int[] backup = Affinity.getAffinity();
        int[] cores = Arrays.copyOfRange(backup, 0, 3);
        Affinity.setAffinity(cores);

        int[] getCores = Affinity.getAffinity();

        assertTrue(getCores.length == 3);
        assertTrue(getCores[0] == backup[0]);
        assertTrue(getCores[1] == backup[1]);
        assertTrue(getCores[2] == backup[2]);

        Affinity.resetAffinity();
    }

    @Test
    public void GetAffinity2() {
        int[] backup = Affinity.getAffinity();

        Affinity.setAffinity();

        int[] affinity = Affinity.getAffinity();
        assertTrue(affinity.length == 1);
        assertTrue(affinity[0] == backup[0]);

        assertTrue(!Affinity.stats().get(affinity[0]).isEmpty());

        Affinity.resetAffinity();
    }

    @Test
    public void GetAffinity3() {
        int[] backup = Affinity.getAffinity();

        Affinity.setAffinity(Arrays.copyOfRange(backup, 0, 2));

        int[] affinity = Affinity.getAffinity();
        assertTrue(affinity.length == 2);

        assertTrue(!Affinity.stats().get(affinity[0]).isEmpty());
        assertTrue(!Affinity.stats().get(affinity[1]).isEmpty());

        Affinity.resetAffinity();
    }

    @Test
    public void SetOmpAffinity() {
        int[] backup = Affinity.getAffinity();
        Affinity.setOmpAffinity();

        int[] cores = Affinity.getAffinity();

        assertTrue(cores.length == 1);
        assertTrue(cores[0] == backup[0]);

        Affinity.resetAffinity();
    }

    @Test
    public void SetOmpAffinityMultiTimes() {
        int[] backup = Affinity.getAffinity();
        oneTime(backup);
        oneTime(backup);
        Affinity.resetAffinity();
    }

    private void oneTime(int[] backup) {
        Affinity.setOmpAffinity();

        int[] ompAffinity = Affinity.getOmpAffinity();
        Arrays.sort(backup);
        Arrays.sort(ompAffinity);

        for (int i = 0; i < backup.length; i++) {
            assertTrue(ompAffinity[i] == backup[i]);
        }
    }
}
