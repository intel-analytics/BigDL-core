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

import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.*;

public class CKKSTest {
  @Test
  public void testSecretSaveLoad() throws IOException {
    CKKS ckks = new CKKS();
    byte[][] secrets = ckks.createSecrets();
    Path tmp = Files.createTempFile("ckksTest", "sect");
    CKKS.saveSecret(secrets, tmp.toString());
    byte[][] loaded = CKKS.loadSecret(tmp.toString());
    for(int i = 0; i < secrets.length; i++) {
      assertArrayEquals(secrets[i], loaded[i]);
    }

  }

  @Test
  public void test() {
    CKKS ckks = new CKKS();
    byte[][] a = ckks.createSecrets();
    System.out.println("java len");
    for (int i = 0; i < 4; i++)
        System.out.println(a[i].length);
    long ckksEncryptor = ckks.createCkksEncryptor(a);
    float[] input = new float[]{0.063364277360961f,
      0.90631252736785f,
      0.22275671223179f,
      0.37516756891273f};
    byte[] enInput = ckks.ckksEncrypt(ckksEncryptor, input);
    float[] deInput = ckks.ckksDecrypt(ckksEncryptor, enInput);
    for (int i = 0; i < 4; i++)
      assertEquals(input[i], deInput[i], 1e-6);


    float[] target = new float[]{1, 1, 0, 1};
    byte[] enTarget = ckks.ckksEncrypt(ckksEncryptor, target);

    long ckksCommon = ckks.createCkksCommon(a);
    System.out.println(ckksCommon);

    byte[][] enResult = ckks.train(ckksCommon, enInput, enTarget);
    float[] loss = ckks.ckksDecrypt(ckksEncryptor, enResult[0]);
    float[] gradInput = ckks.ckksDecrypt(ckksEncryptor, enResult[1]);

    float averageLoss = (loss[0] + loss[1] + loss[2] + loss[3]) / 4;

    assertEquals(averageLoss, 0.5837676, 1.1e-2);
    assertEquals(gradInput[0] / 4, -0.12104105, 1e-2);
    assertEquals(gradInput[1] / 4, -0.07193876, 1e-2);
    assertEquals(gradInput[2] / 4, 0.13886501, 1e-2);
    assertEquals(gradInput[3] / 4, -0.10182324, 1e-2);

    byte[][] enOutput = ckks.sigmoidForward(ckksCommon, enInput);
    float[] output = ckks.ckksDecrypt(ckksEncryptor, enOutput[0]);
    assertEquals(output[0], 0.51583576, 1e-2);
    assertEquals(output[1], 0.712245, 3e-2);
    assertEquals(output[2], 0.55546004, 1e-2);
    assertEquals(output[3], 0.59270704, 2e-2);


    float[] addInput1 = new float[]{0.123f,
      0.321f,
      0.121f,
      0.789f};

    float[] addInput2 = new float[]{0.234f,
      0.432f,
      0.232f,
      0.678f};

    byte[] enAddInput1 = ckks.ckksEncrypt(ckksEncryptor, addInput1);
    byte[] enAddInput2 = ckks.ckksEncrypt(ckksEncryptor, addInput2);
    byte[] enSum = ckks.cadd(ckksCommon, enAddInput1, enAddInput2);
    float[] sum = ckks.ckksDecrypt(ckksEncryptor, enSum);
    for (int i = 0; i < 4; i++)
      assertEquals(sum[i], addInput1[i] + addInput2[i], 1e-5);

  }
}
