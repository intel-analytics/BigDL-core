#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include "nn-fixpoint.h"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"


void TestFC(size_t data_batch, size_t data_channel, size_t filter_num) {
  FixFCOpDesc *desc = FixFCOpCreate(NCHW);
  // init weight
  float *weight;
  std::vector<float> weight_vector;
  weight_vector.resize(filter_num * data_channel);
  std::generate(weight_vector.begin(), weight_vector.end(), []{return 1.0f;});
  weight = weight_vector.data();
  // init bias
  float *bias;
  std::vector<float> bias_vector;
  bias_vector.resize(filter_num);
  std::generate(bias_vector.begin(), bias_vector.end(), []{return 0.0f;});
  bias = bias_vector.data();
  // init data
  float *data;
  std::vector<float> data_vector;
  data_vector.resize(data_batch * data_channel);
  std::generate(data_vector.begin(), data_vector.end(), []{return 1.0f;});
  data = data_vector.data();
  // init out
  float *out;
  size_t out_channel = filter_num;
  std::vector<float> out_vector;
  out_vector.resize(data_batch * out_channel);
  out = out_vector.data();

  FixFCOpSetupFCParameter(desc, filter_num, data_channel, weight, bias, false);
  FixFCOpQuantizeKernel(desc, 64.0f);
  FixFCOpQuantizeData(desc, data_batch, data_channel, data, 127.0f);
  FixFCOpExecuteToDst(desc, out, 0.5f);
  FixFCOpFree(desc);

  for (auto iter = out_vector.begin(); iter < out_vector.end(); ++iter) {
    DOUBLES_EQUAL(*iter, data_channel, 1e-6);
  }
}

TEST_GROUP(FC) {

};

TEST(FC, TEST_FC){
  TestFC(1, 2048, 8192);
  TestFC(2, 4096, 4096);
  TestFC(4, 128, 128);
  TestFC(8, 1023, 1024);
  TestFC(16, 4095, 4095);
  TestFC(32, 31, 31);
  TestFC(64, 4096, 4096);
  TestFC(128, 8192, 8192);
  TestFC(128, 16384, 16384);
  for (size_t i = 0; i < 1024; ++i) {
    TestFC(8, 1024, 1024);
  }
}

int main(int argc, char** argv) {
  return RUN_ALL_TESTS(argc, argv);
}
