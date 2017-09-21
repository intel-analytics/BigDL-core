#include <iostream>
#include <array>
#include <vector>
#include <tuple>
#include <algorithm>
#include "../base.h"
#include "../common.h"
#include "../ops/ops.h"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"

TEST_GROUP(LAYOUT){

};

TEST(LAYOUT, NCHW2NHWC) {
  std::vector<std::tuple<size_t, size_t, size_t>> data;
  data.push_back(std::move(std::make_tuple(32, 128, 13 * 13)));
  data.push_back(std::move(std::make_tuple(32, 256, 23 * 23)));
  data.push_back(std::move(std::make_tuple(32, 512, 33 * 33)));
  data.push_back(std::move(std::make_tuple(128, 64, 43 * 43)));
  data.push_back(std::move(std::make_tuple(128, 128, 7 * 7)));
  data.push_back(std::move(std::make_tuple(128, 256, 15 * 15)));
  data.push_back(std::move(std::make_tuple(128, 512, 31 * 31)));
  for (auto iter = data.begin(); iter < data.end(); ++iter) {
    size_t batch_size = std::get<0>(*iter);
    size_t channels = std::get<0>(*iter);
    size_t spatial_size = std::get<0>(*iter);
    std::vector<float> dst(batch_size * spatial_size * channels);
    std::vector<float> src(batch_size * spatial_size * channels);
    std::generate(src.begin(), src.end(), [] { return static_cast<float>(std::rand()) / RAND_MAX; });
    TransformLayout<float>(NHWC, NCHW, dst.data(), src.data(), batch_size, channels, spatial_size);
    for (size_t b = 0; b < batch_size; ++b) {
      size_t offset = b * channels * spatial_size;
      for (size_t c = 0; c < channels; ++c) {
        for (size_t s = 0; s < spatial_size; ++s) {
          DOUBLES_EQUAL(*(src.data() + offset + c * spatial_size + s), *(dst.data() + offset + s * channels + c), 1e-8);
        }
      }
    }
  }
}

TEST(LAYOUT, NHWC2NCHW) {
  std::vector<std::tuple<size_t, size_t, size_t>> data;
  data.push_back(std::move(std::make_tuple(32, 128, 13 * 13)));
  data.push_back(std::move(std::make_tuple(32, 256, 23 * 23)));
  data.push_back(std::move(std::make_tuple(32, 512, 33 * 33)));
  data.push_back(std::move(std::make_tuple(128, 64, 43 * 43)));
  data.push_back(std::move(std::make_tuple(128, 128, 7 * 7)));
  data.push_back(std::move(std::make_tuple(128, 256, 15 * 15)));
  data.push_back(std::move(std::make_tuple(128, 512, 31 * 31)));
  for (auto iter = data.begin(); iter < data.end(); ++iter) {
    size_t batch_size = std::get<0>(*iter);
    size_t channels = std::get<0>(*iter);
    size_t spatial_size = std::get<0>(*iter);
    std::vector<float> dst(batch_size * spatial_size * channels);
    std::vector<float> src(batch_size * spatial_size * channels);
    std::generate(src.begin(), src.end(), [] { return static_cast<float>(std::rand()) / RAND_MAX; });
    TransformLayout<float>(NCHW, NHWC, dst.data(), src.data(), batch_size, channels, spatial_size);
    for (size_t b = 0; b < batch_size; ++b) {
      size_t offset = b * channels * spatial_size;
      for (size_t c = 0; c < channels; ++c) {
        for (size_t s = 0; s < spatial_size; ++s) {
          DOUBLES_EQUAL(*(src.data() + offset + s * channels + c), *(dst.data() + offset + c * spatial_size + s), 1e-8);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  return RUN_ALL_TESTS(argc, argv);
}
