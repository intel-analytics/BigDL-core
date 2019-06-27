package com.intel.analytics.bigdl.mkl;

import com.intel.analytics.bigdl.mkl.hardware.Affinity;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class AffinityTest {
    @Test
    public void SetSingleAffinity() {
        int[] backup = Affinity.getAffinity();

        Affinity.setAffinity(3);

        int[] cores = Affinity.getAffinity();

        assertTrue(cores.length == 1);
        assertTrue(cores[0] == 3);

        Affinity.setAffinity(backup);
    }

    @Test
    public void SetMultiAffinity() {
        int[] backup = Affinity.getAffinity();
        int[] cores = {1, 2, 3};
        Affinity.setAffinity(cores);

        int[] getCores = Affinity.getAffinity();

        assertTrue(getCores.length == 3);
        assertTrue(getCores[0] == 1);
        assertTrue(getCores[1] == 2);
        assertTrue(getCores[2] == 3);

        Affinity.setAffinity(backup);
    }

    @Test
    public void SetOmpAffinity() {
        int[] backup = Affinity.getAffinity();
        Affinity.setOmpAffinity();

        int[] cores = Affinity.getAffinity();

        assertTrue(cores.length == 1);
        assertTrue(cores[0] == 0);

        Affinity.setAffinity(backup);
    }

}
