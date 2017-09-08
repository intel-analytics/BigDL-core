#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include "nn-fixpoint.h"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"

static size_t GetConvOutSize(size_t in, size_t kernel, size_t stride, size_t pad, size_t dilation) {
  return (in + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
}

void TestConvolutionSaveResult(size_t data_batch, size_t data_channel, size_t data_height, size_t data_width, size_t group, size_t filter_num, size_t filter_height, size_t filter_width, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w) {
  FixConvOpDesc *desc = FixConvOpCreate(NCHW);

  std::vector<float> weight;
  weight.resize(filter_num * data_channel * filter_height * filter_width / group);
  // fill the weight with 1.0f
  std::generate(weight.begin(), weight.end(), []{return 1.0f;});

  // init data
  std::vector<float> data;
  data.resize(data_batch * data_channel * data_height * data_width);
  std::generate(data.begin(), data.end(), []{return 1.0f;});

  // init out
  size_t out_channel = filter_num;
  size_t out_height = GetConvOutSize(data_height, filter_height, stride_h, pad_h, dilation_h);
  size_t out_width = GetConvOutSize(data_width, filter_width, stride_w, pad_w, dilation_w);

  std::vector<float> out;
  out.resize(data_batch * out_channel * out_height * out_width);

  FixConvOpSetupConvParameter(desc, filter_num, data_channel, group, filter_height, filter_width, stride_h, stride_w, dilation_h, dilation_w, pad_h, pad_w, weight.data(), NULL);
  FixConvOpQuantizeKernel(desc, 64.0f);
  FixConvOpQuantizeData(desc, data_batch, data_channel, data_height, data_width, data.data(), 127.0f);
  FixConvOpExecuteToDst(desc, out.data(), 0.5f);
  FixConvOpFree(desc);
  for (auto iter = out.begin(); iter < out.end(); ++iter) {
    DOUBLES_EQUAL(*iter, data_channel * filter_height * filter_width, 1e-6);
  }
}

TEST_GROUP(CONVOLUTION) {

};

TEST(CONVOLUTION, TEST_CONVOLUTION){
  TestConvolutionSaveResult(1, 32, 10, 10, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1);
  TestConvolutionSaveResult(2, 128, 16, 16, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1);
  TestConvolutionSaveResult(4, 128, 16, 16, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1);
  TestConvolutionSaveResult(8, 128, 16, 16, 1, 1, 5, 5, 1, 1, 0, 0, 1, 1);
  TestConvolutionSaveResult(16, 128, 16, 16, 1, 1, 7, 7, 1, 1, 0, 0, 1, 1);
  TestConvolutionSaveResult(32, 128, 16, 16, 1, 1, 11, 11, 1, 1, 0, 0, 1, 1);
}

int main(int argc, char** argv) {
  return RUN_ALL_TESTS(argc, argv);
}
