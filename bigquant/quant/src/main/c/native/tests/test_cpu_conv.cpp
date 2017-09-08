#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include "nn-fixpoint.h"
#include "common.h"
#include "test_common.h"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"
#include "H5Cpp.h"

using namespace H5;
TEST_GROUP(CONVOLUTION) {

};

/*
TEST(CONVOLUTION, TEST_WINOGRAD){
  TestBaseWinograd2x2Kernel3x3ConvolutionSaveResult(16, 512, 16, 16, 1, 512, 3, 3, 1, 1, 0, 0, 1, 1, "winograd");
};
*/

void winograd_v0() {
  for (size_t i = 0; i < 3; ++i)
    TestWinogradImprove2x2Kernel3x3ConvolutionSaveResult(16, 256, 16, 16, 1, 256, 3, 3, 1, 1, 0, 0, 1, 1, "winograd");
  std::cerr << "-----------" << std::endl;
}

void winograd_v1() {
  for (size_t i = 0; i < 4; ++i)
    TestWinogradImprove2x2Kernel3x3ConvolutionSaveResultV1(16, 512, 16, 16, 1, 512, 3, 3, 1, 1, 0, 0, 1, 1, "winograd");
  std::cerr << "-----------" << std::endl;
}

void winograd_v2() {
  for (size_t i = 0; i < 5; ++i)
  TestWinogradImprove2x2Kernel3x3ConvolutionSaveResultV2(64, 256, 16, 16, 1, 256, 3, 3, 1, 1, 0, 0, 1, 1, "winograd");
}

void winograd_v3() {
  for (size_t i = 0; i < 3; ++i)
    TestWinogradImprove2x2Kernel3x3ConvolutionSaveResultV3(16, 256, 16, 16, 1, 256, 3, 3, 1, 1, 0, 0, 1, 1, "winograd");
  std::cerr << "-----------" << std::endl;
}

TEST(CONVOLUTION, TEST_CONVOLUTION){
  /*
  TestConvolutionSaveResult(1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, "case1");
  TestConvolutionSaveResult(2, 1024, 19, 19, 1, 1024, 1, 1, 1, 1, 0, 0, 1, 1, "case2");
  TestConvolutionSaveResult(2, 1024, 19, 19, 1, 126, 3, 3, 1, 1, 1, 1, 1, 1, "case3");
  TestConvolutionSaveResult(2, 1024, 19, 19, 1, 24, 3, 3, 1, 1, 1, 1, 1, 1, "case4");
  TestConvolutionSaveResult(2, 256, 1, 1, 1, 16, 3, 3, 1, 1, 1, 1, 1, 1, "case5");
  TestConvolutionSaveResult(2, 256, 1, 1, 1, 84, 3, 3, 1, 1, 1, 1, 1, 1, "case6");
  TestConvolutionSaveResult(2, 256, 3, 3, 1, 16, 3, 3, 1, 1, 1, 1, 1, 1, "case7");
  TestConvolutionSaveResult(2, 256, 3, 3, 1, 84, 3, 3, 1, 1, 1, 1, 1, 1, "case8");
  TestConvolutionSaveResult(2, 256, 5, 5, 1, 126, 3, 3, 1, 1, 1, 1, 1, 1, "case9");
  TestConvolutionSaveResult(2, 256, 5, 5, 1, 24, 3, 3, 1, 1, 1, 1, 1, 1, "case10");
  TestConvolutionSaveResult(2, 512, 10, 10, 1, 126, 3, 3, 1, 1, 1, 1, 1, 1, "case11");
  TestConvolutionSaveResult(2, 512, 10, 10, 1, 24, 3, 3, 1, 1, 1, 1, 1, 1, "case12");
  TestConvolutionSaveResult(2, 512, 38, 38, 1, 16, 3, 3, 1, 1, 1, 1, 1, 1, "case13");
  TestConvolutionSaveResult(2, 512, 38, 38, 1, 84, 3, 3, 1, 1, 1, 1, 1, 1, "case14");
  */
}

/*
TEST(CONVOLUTION, TEST_TENSOR_CONVOLUTION){
  TestTensorConvolutionSaveResult(1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, "case_tensor_1");
  TestTensorConvolutionSaveResult(2, 1024, 19, 19, 1, 1024, 1, 1, 1, 1, 0, 0, 1, 1, "case_tensor_2");
  TestTensorConvolutionSaveResult(2, 1024, 19, 19, 1, 126, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_3");
  TestTensorConvolutionSaveResult(2, 1024, 19, 19, 1, 24, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_4");
  TestTensorConvolutionSaveResult(2, 256, 1, 1, 1, 16, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_5");
  TestTensorConvolutionSaveResult(2, 256, 1, 1, 1, 84, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_6");
  TestTensorConvolutionSaveResult(2, 256, 3, 3, 1, 16, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_7");
  TestTensorConvolutionSaveResult(2, 256, 3, 3, 1, 84, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_8");
  TestTensorConvolutionSaveResult(2, 256, 5, 5, 1, 126, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_9");
  TestTensorConvolutionSaveResult(2, 256, 5, 5, 1, 24, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_10");
  TestTensorConvolutionSaveResult(2, 512, 10, 10, 1, 126, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_11");
  TestTensorConvolutionSaveResult(2, 512, 10, 10, 1, 24, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_12");
  TestTensorConvolutionSaveResult(2, 512, 38, 38, 1, 16, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_13");
  TestTensorConvolutionSaveResult(2, 512, 38, 38, 1, 84, 3, 3, 1, 1, 1, 1, 1, 1, "case_tensor_14");
}
*/

int main(int argc, char** argv) {
  //winograd_v0();
  winograd_v1();
  //winograd_v2();
  //winograd_v3();

  //return RUN_ALL_TESTS(argc, argv);
}
