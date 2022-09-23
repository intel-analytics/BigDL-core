#include "com_intel_analytics_bigdl_ckks_CKKS.h"
#include "common.h"
#include "ckks_common.h"
#include "kms.h"
#include "data_encryptor.h"

#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    createKms
 * Signature: ()J
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_createSecrets
  (JNIEnv *env, jobject jobj){
      cout<<"In JNI"<<endl;
      LHE_KMS kms;
      kms.generate(COMPUTE_MODE::Train, 16384, sec_level_type::tc128);

      vector<string> secrets = {kms.getEncryptionParamters().str(),
        kms.getPublicKey().str(),
        kms.getRelinearKey().str(),
        kms.getSecretKey().str()};

      jobjectArray ret = (jobjectArray)env->NewObjectArray(4,env->FindClass("[B"),NULL);

      for (int i = 0; i < 4; i++) {
          int n = secrets[i].length();
          jbyteArray byteArray = env->NewByteArray(n);
          jbyte * bytes = (jbyte *)env->GetPrimitiveArrayCritical(byteArray, 0);

          for (int j = 0; j < n; j++) {
                bytes[j] = (jbyte) secrets[i][j];
          }
          env->ReleasePrimitiveArrayCritical(byteArray, bytes, 0);
          env->SetObjectArrayElement(ret, i, byteArray);
      }
      return ret;
  }

/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    createCkksEncryptor
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_createCkksEncryptor
  (JNIEnv *env, jobject jobj, jobjectArray secrets) {
      vector<string> str_secrets;
      int n = env->GetArrayLength(secrets);
      for (int i = 0; i < n; i++) {
        jbyteArray byteArray = (jbyteArray) env->GetObjectArrayElement(secrets, i);
        jbyte *bytes = (jbyte *)env->GetPrimitiveArrayCritical(byteArray, 0);

        int len = env->GetArrayLength(byteArray);
        string s;
        for (int j = 0; j < len; j++) {
            s += bytes[j];
        }
        env->ReleasePrimitiveArrayCritical(byteArray, bytes, 0);
        str_secrets.push_back(s);
      }
      DataEncryptor* de = new DataEncryptor(str_secrets);
      return (jlong)de;
  }

/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    ckksEncrypt
 * Signature: (J[F)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_ckksEncrypt
  (JNIEnv *env, jobject jobj, jlong ckks_encryptor_ptr, jfloatArray jdata) {
      jfloat * cdata = env->GetFloatArrayElements(jdata, 0);
      jsize jsize = env->GetArrayLength(jdata);
      vector<double> vdata;
      for (int i = 0; i < jsize; i++){
          vdata.push_back((double)cdata[i]);
      }

      stringstream ss2 = ((DataEncryptor*)ckks_encryptor_ptr)->encrypt(vdata);

      int n = ss2.str().length();
      const char* sssc = ss2.str().c_str();
      string sss = ss2.str();
      jbyteArray arr = env->NewByteArray(n);
      jbyte *bytes = env->GetByteArrayElements(arr, 0);
      for (int i = 0; i < n; i ++) {
          bytes[i] = (jbyte) sss[i];
      }
      env->SetByteArrayRegion(arr,0,n,bytes);

      return arr;
  }

/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    ckksDecrypt
 * Signature: (J[B)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_ckksDecrypt
  (JNIEnv *env, jobject jobj, jlong ckks_encryptor_ptr, jbyteArray jdata){
      jbyte* buffer = env->GetByteArrayElements(jdata, 0);
      jsize size = env->GetArrayLength(jdata);

      stringstream ss;
      for(int i = 0; i < size; i++) {
          ss << buffer[i];
      }

      SEALContext* context_ = ((DataEncryptor*)ckks_encryptor_ptr)->context_;
      Ciphertext data_encrypted;
      data_encrypted.load(*context_, ss);
      vector<double> dd = ((DataEncryptor*)ckks_encryptor_ptr)->decrypt(data_encrypted);

      jfloatArray result = env->NewFloatArray(dd.size());
      jfloat re_array[dd.size()];

      for(int i = 0; i < dd.size(); i++) {
          float d = (float) dd[i];
          re_array[i] = (jfloat) d;
      }
      env->SetFloatArrayRegion(result, 0, dd.size(), re_array);

      return result;
  }

/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    createCkksCommonInstance
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_createCkksCommonInstance
  (JNIEnv *env, jobject jobj, jobjectArray secrets){
      vector<string> str_secrets;
      int n = env->GetArrayLength(secrets);
      for (int i = 0; i < n; i++) {
        jbyteArray byteArray = (jbyteArray) env->GetObjectArrayElement(secrets, i);
        jbyte *bytes = (jbyte *)env->GetPrimitiveArrayCritical(byteArray, 0);

        int len = env->GetArrayLength(byteArray);
        string s;
        for (int j = 0; j < len; j++) {
            s += bytes[j];
        }
        env->ReleasePrimitiveArrayCritical(byteArray, bytes, 0);
        str_secrets.push_back(s);
      }
      return (jlong) new CKKS_Common(str_secrets);
  }

/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    train
 * Signature: (J[B[B)[B
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_train
  (JNIEnv * env, jobject jobj, jlong ckks_common_ptr, jbyteArray jinput, jbyteArray jtarget){
      CKKS_Common* ckks_common_ = (CKKS_Common*)ckks_common_ptr;
      SEALContext* context_ = ((CKKS_Common*)ckks_common_ptr)->context_;

      stringstream ss;
      Ciphertext x_encrypted;
      ss.str("");
      jbyte *cinput = (jbyte *)env->GetPrimitiveArrayCritical(jinput, 0);

      int output_len = env->GetArrayLength(jinput);
      for (int j = 0; j < output_len; j++) {
          ss << cinput[j];
      }
      env->ReleasePrimitiveArrayCritical(jinput, cinput, 0);
      x_encrypted.load(*context_, ss);

      Ciphertext y_encrypted;
      ss.str("");
      jbyte *ctarget = (jbyte *)env->GetPrimitiveArrayCritical(jtarget, 0);
      int target_len = env->GetArrayLength(jtarget);
      for (int j = 0; j < target_len; j++) {
          ss << ctarget[j];
      }
      env->ReleasePrimitiveArrayCritical(jtarget, ctarget, 0);
      y_encrypted.load(*context_, ss);

      Ciphertext pred_encrypted =
          ckks_common_->computeSigmoid7thDegree(x_encrypted);
      Ciphertext loss_encrypted =
          ckks_common_->computeBCE(pred_encrypted, y_encrypted);
      Ciphertext backwards_encrypted =
          ckks_common_->computeBackwards(pred_encrypted, y_encrypted);

      jobjectArray ret = (jobjectArray)env->NewObjectArray(2,env->FindClass("[B"),NULL);

      ss.str("");
      loss_encrypted.save(ss);
      string loss = ss.str();

      int loss_n = loss.length();
      jbyteArray loss_byteArray = env->NewByteArray(loss_n);
      jbyte * loss_bytes = (jbyte *)env->GetPrimitiveArrayCritical(loss_byteArray, 0);

      for (int j = 0; j < loss_n; j++) {
            loss_bytes[j] = (jbyte) loss[j];
      }
      env->ReleasePrimitiveArrayCritical(loss_byteArray, loss_bytes, 0);
      env->SetObjectArrayElement(ret, 0, loss_byteArray);

      ss.str("");
      backwards_encrypted.save(ss);
      string backwards = ss.str();

      int back_n = backwards.length();
      jbyteArray back_byteArray = env->NewByteArray(back_n);
      jbyte * back_bytes = (jbyte *)env->GetPrimitiveArrayCritical(back_byteArray, 0);

      for (int j = 0; j < back_n; j++) {
            back_bytes[j] = (jbyte) backwards[j];
      }
      env->ReleasePrimitiveArrayCritical(back_byteArray, back_bytes, 0);
      env->SetObjectArrayElement(ret, 1, back_byteArray);

      return ret;
  }

/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    backward
 * Signature: (J[B[B)[[B
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_backward
  (JNIEnv * env, jobject jobj, jlong ckks_common_ptr, jbyteArray jinput, jbyteArray jtarget){
      CKKS_Common* ckks_common_ = (CKKS_Common*)ckks_common_ptr;
      SEALContext* context_ = ((CKKS_Common*)ckks_common_ptr)->context_;

      stringstream ss;
      Ciphertext x_encrypted;
      ss.str("");
      jbyte *cinput = (jbyte *)env->GetPrimitiveArrayCritical(jinput, 0);

      int output_len = env->GetArrayLength(jinput);
      for (int j = 0; j < output_len; j++) {
          ss << cinput[j];
      }
      env->ReleasePrimitiveArrayCritical(jinput, cinput, 0);
      x_encrypted.load(*context_, ss);

      Ciphertext y_encrypted;
      ss.str("");
      jbyte *ctarget = (jbyte *)env->GetPrimitiveArrayCritical(jtarget, 0);
      int target_len = env->GetArrayLength(jtarget);
      for (int j = 0; j < target_len; j++) {
          ss << ctarget[j];
      }
      env->ReleasePrimitiveArrayCritical(jtarget, ctarget, 0);
      y_encrypted.load(*context_, ss);

      Ciphertext loss_encrypted =
          ckks_common_->computeBCE(x_encrypted, y_encrypted);
      Ciphertext backwards_encrypted =
          ckks_common_->computeBackwards(x_encrypted, y_encrypted);

      jobjectArray ret = (jobjectArray)env->NewObjectArray(2,env->FindClass("[B"),NULL);

      ss.str("");
      loss_encrypted.save(ss);
      string loss = ss.str();

      int loss_n = loss.length();
      jbyteArray loss_byteArray = env->NewByteArray(loss_n);
      jbyte * loss_bytes = (jbyte *)env->GetPrimitiveArrayCritical(loss_byteArray, 0);

      for (int j = 0; j < loss_n; j++) {
            loss_bytes[j] = (jbyte) loss[j];
      }
      env->ReleasePrimitiveArrayCritical(loss_byteArray, loss_bytes, 0);
      env->SetObjectArrayElement(ret, 0, loss_byteArray);

      ss.str("");
      backwards_encrypted.save(ss);
      string backwards = ss.str();

      int back_n = backwards.length();
      jbyteArray back_byteArray = env->NewByteArray(back_n);
      jbyte * back_bytes = (jbyte *)env->GetPrimitiveArrayCritical(back_byteArray, 0);

      for (int j = 0; j < back_n; j++) {
            back_bytes[j] = (jbyte) backwards[j];
      }
      env->ReleasePrimitiveArrayCritical(back_byteArray, back_bytes, 0);
      env->SetObjectArrayElement(ret, 1, back_byteArray);

      return ret;
  }

/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    sigmoidForward
 * Signature: (J[B)[[B
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_sigmoidForward
  (JNIEnv * env, jobject jobj, jlong ckks_common_ptr, jbyteArray jinput){
      CKKS_Common* ckks_common_ = (CKKS_Common*)ckks_common_ptr;
      SEALContext* context_ = ((CKKS_Common*)ckks_common_ptr)->context_;

      stringstream ss;
      Ciphertext x_encrypted;
      ss.str("");
      jbyte *cinput = (jbyte *)env->GetPrimitiveArrayCritical(jinput, 0);

      int output_len = env->GetArrayLength(jinput);
      for (int j = 0; j < output_len; j++) {
          ss << cinput[j];
      }
      env->ReleasePrimitiveArrayCritical(jinput, cinput, 0);
      x_encrypted.load(*context_, ss);

      Ciphertext pred_encrypted =
          ckks_common_->computeSigmoid7thDegree(x_encrypted);

      jobjectArray ret = (jobjectArray)env->NewObjectArray(1,env->FindClass("[B"),NULL);

      ss.str("");
      pred_encrypted.save(ss);
      string pred = ss.str();

      int pred_n = pred.length();
      jbyteArray pred_byteArray = env->NewByteArray(pred_n);
      jbyte * pred_bytes = (jbyte *)env->GetPrimitiveArrayCritical(pred_byteArray, 0);

      for (int j = 0; j < pred_n; j++) {
            pred_bytes[j] = (jbyte) pred[j];
      }
      env->ReleasePrimitiveArrayCritical(pred_byteArray, pred_bytes, 0);
      env->SetObjectArrayElement(ret, 0, pred_byteArray);

      return ret;
  }

/*
 * Class:     com_intel_analytics_bigdl_ckks_CKKS
 * Method:    cadd
 * Signature: (J[B[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_intel_analytics_bigdl_ckks_CKKS_cadd
  (JNIEnv * env, jobject jobj, jlong ckks_common_ptr, jbyteArray jinput1, jbyteArray jinput2){
      CKKS_Common* ckks_common_ = (CKKS_Common*)ckks_common_ptr;
      SEALContext* context_ = ((CKKS_Common*)ckks_common_ptr)->context_;

      vector<Ciphertext> inputs_encrypted;

      stringstream ss;
      Ciphertext x1_encrypted;
      ss.str("");
      jbyte *cinput1 = (jbyte *)env->GetPrimitiveArrayCritical(jinput1, 0);
      int output1_len = env->GetArrayLength(jinput1);
      for (int j = 0; j < output1_len; j++) {
          ss << cinput1[j];
      }
      env->ReleasePrimitiveArrayCritical(jinput1, cinput1, 0);
      x1_encrypted.load(*context_, ss);

      inputs_encrypted.push_back(x1_encrypted);

      Ciphertext x2_encrypted;
      ss.str("");
      jbyte *cinput2 = (jbyte *)env->GetPrimitiveArrayCritical(jinput2, 0);
      int output2_len = env->GetArrayLength(jinput2);
      for (int j = 0; j < output2_len; j++) {
          ss << cinput2[j];
      }
      env->ReleasePrimitiveArrayCritical(jinput2, cinput2, 0);
      x2_encrypted.load(*context_, ss);

      inputs_encrypted.push_back(x2_encrypted);

      Ciphertext sum_encrypted =
          ckks_common_->computeCAddTable(inputs_encrypted);


      ss.str("");
      sum_encrypted.save(ss);
      string sum = ss.str();

      int sum_n = sum.length();
      jbyteArray sum_byteArray = env->NewByteArray(sum_n);
      jbyte * sum_bytes = (jbyte *)env->GetPrimitiveArrayCritical(sum_byteArray, 0);

      for (int j = 0; j < sum_n; j++) {
            sum_bytes[j] = (jbyte) sum[j];
      }
      env->ReleasePrimitiveArrayCritical(sum_byteArray, sum_bytes, 0);

      return sum_byteArray;
  }

#ifdef __cplusplus
}
#endif
