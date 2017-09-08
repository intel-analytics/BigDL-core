#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include "../base.h"
#include "../common.h"
#include "../ops/ops.h"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"

TEST_GROUP(Find_Extreme) {

};

TEST(Find_Extreme, Find_Extreme_Float32) {
  std::vector<size_t> length = {2, 3, 4, 8, 9, 10, 16, 17, 30, 31, 32, 63, 64, 65, 127, 128, 129, 255, 255, 257, 511, 512, 513};
  for (auto it = length.begin(); it < length.end(); ++it) {
    float max, min;
    std::vector<float> data(*it);
    std::generate(data.begin(), data.end(), []{return static_cast<float>(std::rand()) / RAND_MAX;});
    FindMinMaxValue<float>(data.data(), data.size(), min, max);
    auto minmax = std::minmax_element(data.begin(), data.end());
    auto min_ref = *(minmax.first);
    auto max_ref = *(minmax.second);
    DOUBLES_EQUAL(max_ref, max, 1e-6);
    DOUBLES_EQUAL(min_ref, min, 1e-6);
  }
}


TEST(Find_Extreme, Find_Extreme_Float64) {
  std::vector<size_t> length = {2, 3, 4, 8, 9, 10, 16, 17, 30, 31, 32, 63, 64, 65, 127, 128, 129, 255, 255, 257, 511, 512, 513};
  for (auto it = length.begin(); it < length.end(); ++it) {
    double max, min;
    std::vector<double> data(*it);
    std::generate(data.begin(), data.end(), []{return static_cast<double>(std::rand()) / RAND_MAX;});
    FindMinMaxValue<double>(data.data(), data.size(), min, max);
    auto minmax = std::minmax_element(data.begin(), data.end());
    auto min_ref = *(minmax.first);
    auto max_ref = *(minmax.second);
    DOUBLES_EQUAL(max_ref, max, 1e-6);
    DOUBLES_EQUAL(min_ref, min, 1e-6);
  }
}

int main(int argc, char** argv) {
  return RUN_ALL_TESTS(argc, argv);
}
