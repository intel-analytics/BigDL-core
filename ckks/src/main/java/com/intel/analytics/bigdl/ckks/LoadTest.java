package com.intel.analytics.bigdl.ckks;

public class LoadTest {
  public static void main(String[] args) {
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


    float[] target = new float[]{1, 1, 0, 1};
    byte[] enTarget = ckks.ckksEncrypt(ckksEncryptor, target);

    long ckksCommon = ckks.createCkksCommonInstance(a);
    System.out.println(ckksCommon);

    byte[][] enResult = ckks.train(ckksCommon, enInput, enTarget);
    float[] loss = ckks.ckksDecrypt(ckksEncryptor, enResult[0]);
    float[] gradInput = ckks.ckksDecrypt(ckksEncryptor, enResult[1]);

    float averageLoss = (loss[0] + loss[1] + loss[2] + loss[3]) / 4;

    byte[][] enOutput = ckks.sigmoidForward(ckksCommon, enInput);
    float[] output = ckks.ckksDecrypt(ckksEncryptor, enOutput[0]);
  }
}
