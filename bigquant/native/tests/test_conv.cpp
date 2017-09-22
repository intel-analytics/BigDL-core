#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include "bigquant.h"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"

static size_t GetConvOutSize(size_t in, size_t kernel, size_t stride, size_t pad, size_t dilation) {
  return (in + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
}

void TestConvolutionDesc(size_t data_batch, size_t data_channel, size_t data_height, size_t data_width, size_t group,
                         size_t filter_num, size_t filter_height, size_t filter_width, size_t stride_h, size_t stride_w,
                         size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, LAYOUT layout) {
  QuantizedConvOp* desc = QuantizedConvOpCreate();

  std::vector<float> weight;
  weight.resize(filter_num * data_channel * filter_height * filter_width / group);
  // fill the weight with 1.0f
  std::generate(weight.begin(), weight.end(), [] { return 1.0f; });

  // init data
  std::vector<float> data;
  data.resize(data_batch * data_channel * data_height * data_width);
  std::generate(data.begin(), data.end(), [] { return 1.0f; });

  // init out
  size_t out_channel = filter_num;
  size_t out_height = GetConvOutSize(data_height, filter_height, stride_h, pad_h, dilation_h);
  size_t out_width = GetConvOutSize(data_width, filter_width, stride_w, pad_w, dilation_w);

  std::vector<float> out;
  out.resize(data_batch * out_channel * out_height * out_width);

  QuantizedConvOpSetupConvParameter(desc, layout, filter_num, data_channel, group, filter_height, filter_width,
                                    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, 0, SHUFFLE_CONV);
  QuantizedConvOpInitWeight(desc, weight.data());
  QuantizedConvOpExecute(desc, out.data(), data.data(), NULL, data_batch, data_channel, data_height, data_width);
  QuantizedConvOpFree(desc);
  for (auto iter = out.begin(); iter < out.end(); ++iter) {
    DOUBLES_EQUAL(*iter, data_channel * filter_height * filter_width, 1e-6);
  }
}

void TestConvolutionTensor(size_t data_batch, size_t data_channel, size_t data_height, size_t data_width, size_t group,
                           size_t filter_num, size_t filter_height, size_t filter_width, size_t stride_h,
                           size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w,
                           LAYOUT layout) {
  // init weight
  float* weight;

  std::vector<float> weight_vector(filter_num * data_channel * filter_height * filter_width / group, 1);
  weight = weight_vector.data();
  // init data
  float* data;
  std::vector<float> data_vector(data_batch * data_channel * data_height * data_width, 1);
  data = data_vector.data();
  // init out
  QuantizedTensorDesc* kernel_tensor = new QuantizedTensorDesc();
  QuantizedTensorDesc* data_tensor = new QuantizedTensorDesc();
  FPTensorDesc* kernel_sum_tensor = new FPTensorDesc();
  QuantizedConvKernelDescInit(kernel_tensor, filter_num, data_channel, filter_height, filter_height);
  QuantizedConvKernelInit(kernel_tensor, weight, filter_num, data_channel, filter_height, filter_width, 64.0f, layout);
  QuantizedConvKernelSumDescInit(kernel_sum_tensor, filter_num);
  QuantizedConvKernelSumInit(kernel_sum_tensor, weight, filter_num, data_channel, filter_height, filter_width);
  size_t out_channel = filter_num;
  size_t out_height = GetConvOutSize(data_height, filter_height, stride_h, pad_h, dilation_h);
  size_t out_width = GetConvOutSize(data_width, filter_width, stride_w, pad_w, dilation_w);
  std::vector<float> out(data_batch * out_channel * out_height * out_width, 0.0f);
  QuantizedConvDataDescInit(data_tensor, data_channel, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w,
                            dilation_h, dilation_w, data_batch, data_height, data_width);
  QuantizedConvDataInit(data_tensor, data, data_channel, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w,
                        dilation_h, dilation_w, data_batch, data_height, data_width, 127.0, layout);
  MixPrecisionGEMM(layout, reinterpret_cast<int8_t*>(kernel_tensor->data),
                   reinterpret_cast<uint8_t*>(data_tensor->data), out.data(), kernel_tensor->shape[0],
                   data_tensor->shape[0], data_tensor->shape[1], reinterpret_cast<float*>(kernel_tensor->ratio),
                   reinterpret_cast<float*>(data_tensor->ratio), reinterpret_cast<float*>(kernel_sum_tensor->data),
                   reinterpret_cast<float*>(data_tensor->min), NULL, data_batch, filter_num, out_height, out_width, 0.5,
                   kernel_tensor->shape[0] - kernel_tensor->ori_shape[0],
                   data_tensor->shape[0] - data_tensor->ori_shape[0]);
  for (auto iter = out.begin(); iter < out.end(); ++iter) {
    DOUBLES_EQUAL(*iter, data_channel * filter_height * filter_width, 1e-6);
  }
  FreeQuantizedTensor(kernel_tensor);
  FreeQuantizedTensor(data_tensor);
  FreeFPTensor(kernel_sum_tensor);
  delete kernel_tensor;
  delete data_tensor;
  delete kernel_sum_tensor;
}

TEST_GROUP(CONVOLUTION){

};

TEST(CONVOLUTION, TEST_NHWC_CONVOLUTION) {
  TestConvolutionDesc(1, 1, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(1, 1, 10, 10, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(1, 1, 38, 38, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(1, 1, 38, 38, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(1, 1, 38, 38, 1, 84, 1, 1, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(1, 1, 38, 38, 1, 84, 3, 3, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(1, 32, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(1, 32, 10, 10, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(2, 128, 16, 16, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(4, 128, 16, 16, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(8, 128, 16, 16, 1, 1, 5, 5, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(16, 128, 16, 16, 1, 1, 7, 7, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionDesc(32, 128, 16, 16, 1, 1, 11, 11, 1, 1, 0, 0, 1, 1, NHWC);
}

TEST(CONVOLUTION, TEST_NCHW_CONVOLUTION) {
  TestConvolutionDesc(1, 1, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(1, 1, 10, 10, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(1, 1, 38, 38, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(1, 1, 38, 38, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(1, 1, 38, 38, 1, 84, 1, 1, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(1, 1, 38, 38, 1, 84, 3, 3, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(1, 32, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(1, 32, 10, 10, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(2, 128, 16, 16, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(4, 128, 16, 16, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(8, 128, 16, 16, 1, 1, 5, 5, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(16, 128, 16, 16, 1, 1, 7, 7, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionDesc(32, 128, 16, 16, 1, 1, 11, 11, 1, 1, 0, 0, 1, 1, NCHW);
}

TEST(CONVOLUTION, TEST_NHWC_CONVOLUTION_TENSOR) {
  TestConvolutionTensor(1, 1, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(1, 1, 10, 10, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(1, 1, 38, 38, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(1, 1, 38, 38, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(1, 1, 38, 38, 1, 84, 1, 1, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(1, 1, 38, 38, 1, 84, 3, 3, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(1, 32, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(1, 32, 10, 10, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(2, 128, 16, 16, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(4, 128, 16, 16, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(8, 128, 16, 16, 1, 1, 5, 5, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(16, 128, 16, 16, 1, 1, 7, 7, 1, 1, 0, 0, 1, 1, NHWC);
  TestConvolutionTensor(32, 128, 16, 16, 1, 1, 11, 11, 1, 1, 0, 0, 1, 1, NHWC);
}

TEST(CONVOLUTION, TEST_NCHW_CONVOLUTION_TENSOR) {
  TestConvolutionTensor(1, 1, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(1, 1, 10, 10, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(1, 1, 38, 38, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(1, 1, 38, 38, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(1, 1, 38, 38, 1, 84, 1, 1, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(1, 1, 38, 38, 1, 84, 3, 3, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(1, 32, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(1, 32, 10, 10, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(2, 128, 16, 16, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(4, 128, 16, 16, 1, 1, 3, 3, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(8, 128, 16, 16, 1, 1, 5, 5, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(16, 128, 16, 16, 1, 1, 7, 7, 1, 1, 0, 0, 1, 1, NCHW);
  TestConvolutionTensor(32, 128, 16, 16, 1, 1, 11, 11, 1, 1, 0, 0, 1, 1, NCHW);
}

int main(int argc, char** argv) {
  return RUN_ALL_TESTS(argc, argv);
}
