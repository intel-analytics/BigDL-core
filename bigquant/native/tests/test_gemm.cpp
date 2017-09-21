#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cblas.h>
#include "../base.h"
#include "../common.h"
#include "../ops/ops.h"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"

TEST_GROUP(GEMM){

};

TEST(GEMM, ConstMixPrecisionGEMM) {
  std::vector<std::tuple<size_t, size_t, size_t>> data;
  //  some random cases
  data.push_back(std::move(std::make_tuple(1, 1, 15)));
  data.push_back(std::move(std::make_tuple(2, 1, 15)));
  data.push_back(std::move(std::make_tuple(2, 2, 15)));
  data.push_back(std::move(std::make_tuple(2, 2, 16)));
  data.push_back(std::move(std::make_tuple(3, 4, 15)));
  data.push_back(std::move(std::make_tuple(4, 4, 8)));
  data.push_back(std::move(std::make_tuple(4, 4, 31)));
  data.push_back(std::move(std::make_tuple(4, 4, 32)));
  data.push_back(std::move(std::make_tuple(4, 8, 8)));
  data.push_back(std::move(std::make_tuple(4, 8, 16)));
  data.push_back(std::move(std::make_tuple(4, 5, 32)));
  data.push_back(std::move(std::make_tuple(5, 5, 32)));
  data.push_back(std::move(std::make_tuple(4, 4, 64)));
  data.push_back(std::move(std::make_tuple(32, 1024, 1024)));
  data.push_back(std::move(std::make_tuple(32, 4096, 4096)));
  data.push_back(std::move(std::make_tuple(32, 311, 393)));
  data.push_back(std::move(std::make_tuple(127, 311, 393)));
  data.push_back(std::move(std::make_tuple(128, 4096, 4096)));
  data.push_back(std::move(std::make_tuple(128, 2048, 2048)));
  data.push_back(std::move(std::make_tuple(128, 1024, 1024)));
  data.push_back(std::move(std::make_tuple(128, 256, 256)));
  data.push_back(std::move(std::make_tuple(128, 128, 128)));
  data.push_back(std::move(std::make_tuple(255, 255, 255)));
  data.push_back(std::move(std::make_tuple(255, 127, 127)));
  data.push_back(std::move(std::make_tuple(255, 127, 127)));
  data.push_back(std::move(std::make_tuple(1023, 1023, 1023)));
  data.push_back(std::move(std::make_tuple(4090, 4090, 4090)));
  data.push_back(std::move(std::make_tuple(4096, 4096, 4096)));
  data.push_back(std::move(std::make_tuple(8192, 8192, 8192)));
  //  Googlenet v1
  data.push_back(std::move(std::make_tuple(64, 12544, 152)));
  data.push_back(std::move(std::make_tuple(64, 3136, 64)));
  data.push_back(std::move(std::make_tuple(192, 3136, 576)));
  data.push_back(std::move(std::make_tuple(64, 784, 192)));
  data.push_back(std::move(std::make_tuple(96, 784, 192)));
  data.push_back(std::move(std::make_tuple(128, 784, 864)));
  data.push_back(std::move(std::make_tuple(16, 784, 192)));
  data.push_back(std::move(std::make_tuple(32, 784, 400)));
  data.push_back(std::move(std::make_tuple(32, 784, 192)));
  data.push_back(std::move(std::make_tuple(128, 784, 256)));
  data.push_back(std::move(std::make_tuple(128, 784, 256)));
  data.push_back(std::move(std::make_tuple(192, 784, 1152)));
  data.push_back(std::move(std::make_tuple(32, 784, 256)));
  data.push_back(std::move(std::make_tuple(96, 784, 800)));
  data.push_back(std::move(std::make_tuple(64, 784, 256)));
  data.push_back(std::move(std::make_tuple(192, 200, 480)));
  data.push_back(std::move(std::make_tuple(96, 200, 480)));
  data.push_back(std::move(std::make_tuple(208, 200, 864)));
  data.push_back(std::move(std::make_tuple(16, 200, 480)));
  data.push_back(std::move(std::make_tuple(48, 200, 400)));
  data.push_back(std::move(std::make_tuple(64, 200, 480)));
  data.push_back(std::move(std::make_tuple(160, 200, 512)));
  data.push_back(std::move(std::make_tuple(112, 200, 512)));
  data.push_back(std::move(std::make_tuple(224, 200, 1008)));
  data.push_back(std::move(std::make_tuple(24, 200, 512)));
  data.push_back(std::move(std::make_tuple(64, 200, 600)));
  data.push_back(std::move(std::make_tuple(64, 200, 512)));
  data.push_back(std::move(std::make_tuple(128, 200, 512)));
  data.push_back(std::move(std::make_tuple(128, 200, 512)));
  data.push_back(std::move(std::make_tuple(256, 200, 1152)));
  data.push_back(std::move(std::make_tuple(24, 200, 512)));
  data.push_back(std::move(std::make_tuple(64, 200, 600)));
  data.push_back(std::move(std::make_tuple(64, 200, 512)));
  data.push_back(std::move(std::make_tuple(112, 200, 512)));
  data.push_back(std::move(std::make_tuple(144, 200, 512)));
  data.push_back(std::move(std::make_tuple(288, 200, 1296)));
  data.push_back(std::move(std::make_tuple(32, 200, 512)));
  data.push_back(std::move(std::make_tuple(64, 200, 800)));
  data.push_back(std::move(std::make_tuple(64, 200, 512)));
  data.push_back(std::move(std::make_tuple(256, 200, 528)));
  data.push_back(std::move(std::make_tuple(160, 200, 528)));
  data.push_back(std::move(std::make_tuple(320, 200, 1440)));
  data.push_back(std::move(std::make_tuple(32, 200, 528)));
  data.push_back(std::move(std::make_tuple(128, 200, 800)));
  data.push_back(std::move(std::make_tuple(128, 200, 528)));
  data.push_back(std::move(std::make_tuple(256, 56, 832)));
  data.push_back(std::move(std::make_tuple(160, 56, 832)));
  data.push_back(std::move(std::make_tuple(320, 56, 1440)));
  data.push_back(std::move(std::make_tuple(32, 56, 832)));
  data.push_back(std::move(std::make_tuple(128, 56, 800)));
  data.push_back(std::move(std::make_tuple(128, 56, 832)));
  data.push_back(std::move(std::make_tuple(384, 56, 832)));
  data.push_back(std::move(std::make_tuple(192, 56, 832)));
  data.push_back(std::move(std::make_tuple(384, 56, 1728)));
  data.push_back(std::move(std::make_tuple(48, 56, 832)));
  data.push_back(std::move(std::make_tuple(128, 56, 1200)));
  data.push_back(std::move(std::make_tuple(128, 56, 832)));
  for (auto it = data.begin(); it < data.end(); ++it) {
    size_t m = std::get<0>(*it);
    size_t n = std::get<1>(*it);
    size_t k = std::get<2>(*it);
    std::vector<int8_t> a(m * k);
    std::generate(a.begin(), a.end(), [] { return 1; });
    std::vector<float> a_ref(m * k);
    std::generate(a_ref.begin(), a_ref.end(), [] { return 1.0f; });
    std::vector<uint8_t> b(n * k);
    std::generate(b.begin(), b.end(), [] { return 2; });
    std::vector<float> b_ref(n * k);
    std::generate(b_ref.begin(), b_ref.end(), [] { return 2.0f; });
    std::vector<int> c(m * n);
    std::vector<float> c_ref(m * n);
    MixPrecisionGemm(static_cast<ORDER>(101), static_cast<TRANSPOSE>(111), static_cast<TRANSPOSE>(112), m, n, k,
                     a.data(), k, b.data(), k, c.data(), n, 0.5);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a_ref.data(), k, b_ref.data(), k, 0.0f,
                c_ref.data(), n);
    for (int i = 0; i < c.size(); ++i) {
      DOUBLES_EQUAL(c_ref[i], static_cast<float>(c[i]), 1e-10);
    }
  }
}

TEST(GEMM, RandMixPrecisionGEMM) {
  const size_t threshold = 32.0;
  std::vector<std::tuple<size_t, size_t, size_t>> data;
  //  some random cases
  data.push_back(std::move(std::make_tuple(1, 1, 15)));
  data.push_back(std::move(std::make_tuple(2, 1, 15)));
  data.push_back(std::move(std::make_tuple(2, 2, 15)));
  data.push_back(std::move(std::make_tuple(2, 2, 16)));
  data.push_back(std::move(std::make_tuple(3, 4, 15)));
  data.push_back(std::move(std::make_tuple(4, 4, 8)));
  data.push_back(std::move(std::make_tuple(4, 4, 31)));
  data.push_back(std::move(std::make_tuple(4, 4, 32)));
  data.push_back(std::move(std::make_tuple(4, 8, 8)));
  data.push_back(std::move(std::make_tuple(4, 8, 16)));
  data.push_back(std::move(std::make_tuple(4, 5, 32)));
  data.push_back(std::move(std::make_tuple(5, 5, 32)));
  data.push_back(std::move(std::make_tuple(4, 4, 64)));
  data.push_back(std::move(std::make_tuple(32, 1024, 1024)));
  data.push_back(std::move(std::make_tuple(32, 4096, 4096)));
  data.push_back(std::move(std::make_tuple(32, 311, 393)));
  data.push_back(std::move(std::make_tuple(127, 311, 393)));
  data.push_back(std::move(std::make_tuple(128, 4096, 4096)));
  data.push_back(std::move(std::make_tuple(128, 2048, 2048)));
  data.push_back(std::move(std::make_tuple(128, 1024, 1024)));
  data.push_back(std::move(std::make_tuple(128, 256, 256)));
  data.push_back(std::move(std::make_tuple(128, 128, 128)));
  data.push_back(std::move(std::make_tuple(255, 255, 255)));
  data.push_back(std::move(std::make_tuple(255, 127, 127)));
  data.push_back(std::move(std::make_tuple(255, 127, 127)));
  data.push_back(std::move(std::make_tuple(1023, 1023, 1023)));
  data.push_back(std::move(std::make_tuple(4090, 4090, 4090)));
  data.push_back(std::move(std::make_tuple(4096, 4096, 4096)));
  data.push_back(std::move(std::make_tuple(8192, 8192, 8192)));
  //  Googlenet v1
  data.push_back(std::move(std::make_tuple(64, 12544, 152)));
  data.push_back(std::move(std::make_tuple(64, 3136, 64)));
  data.push_back(std::move(std::make_tuple(192, 3136, 576)));
  data.push_back(std::move(std::make_tuple(64, 784, 192)));
  data.push_back(std::move(std::make_tuple(96, 784, 192)));
  data.push_back(std::move(std::make_tuple(128, 784, 864)));
  data.push_back(std::move(std::make_tuple(16, 784, 192)));
  data.push_back(std::move(std::make_tuple(32, 784, 400)));
  data.push_back(std::move(std::make_tuple(32, 784, 192)));
  data.push_back(std::move(std::make_tuple(128, 784, 256)));
  data.push_back(std::move(std::make_tuple(128, 784, 256)));
  data.push_back(std::move(std::make_tuple(192, 784, 1152)));
  data.push_back(std::move(std::make_tuple(32, 784, 256)));
  data.push_back(std::move(std::make_tuple(96, 784, 800)));
  data.push_back(std::move(std::make_tuple(64, 784, 256)));
  data.push_back(std::move(std::make_tuple(192, 200, 480)));
  data.push_back(std::move(std::make_tuple(96, 200, 480)));
  data.push_back(std::move(std::make_tuple(208, 200, 864)));
  data.push_back(std::move(std::make_tuple(16, 200, 480)));
  data.push_back(std::move(std::make_tuple(48, 200, 400)));
  data.push_back(std::move(std::make_tuple(64, 200, 480)));
  data.push_back(std::move(std::make_tuple(160, 200, 512)));
  data.push_back(std::move(std::make_tuple(112, 200, 512)));
  data.push_back(std::move(std::make_tuple(224, 200, 1008)));
  data.push_back(std::move(std::make_tuple(24, 200, 512)));
  data.push_back(std::move(std::make_tuple(64, 200, 600)));
  data.push_back(std::move(std::make_tuple(64, 200, 512)));
  data.push_back(std::move(std::make_tuple(128, 200, 512)));
  data.push_back(std::move(std::make_tuple(128, 200, 512)));
  data.push_back(std::move(std::make_tuple(256, 200, 1152)));
  data.push_back(std::move(std::make_tuple(24, 200, 512)));
  data.push_back(std::move(std::make_tuple(64, 200, 600)));
  data.push_back(std::move(std::make_tuple(64, 200, 512)));
  data.push_back(std::move(std::make_tuple(112, 200, 512)));
  data.push_back(std::move(std::make_tuple(144, 200, 512)));
  data.push_back(std::move(std::make_tuple(288, 200, 1296)));
  data.push_back(std::move(std::make_tuple(32, 200, 512)));
  data.push_back(std::move(std::make_tuple(64, 200, 800)));
  data.push_back(std::move(std::make_tuple(64, 200, 512)));
  data.push_back(std::move(std::make_tuple(256, 200, 528)));
  data.push_back(std::move(std::make_tuple(160, 200, 528)));
  data.push_back(std::move(std::make_tuple(320, 200, 1440)));
  data.push_back(std::move(std::make_tuple(32, 200, 528)));
  data.push_back(std::move(std::make_tuple(128, 200, 800)));
  data.push_back(std::move(std::make_tuple(128, 200, 528)));
  data.push_back(std::move(std::make_tuple(256, 56, 832)));
  data.push_back(std::move(std::make_tuple(160, 56, 832)));
  data.push_back(std::move(std::make_tuple(320, 56, 1440)));
  data.push_back(std::move(std::make_tuple(32, 56, 832)));
  data.push_back(std::move(std::make_tuple(128, 56, 800)));
  data.push_back(std::move(std::make_tuple(128, 56, 832)));
  data.push_back(std::move(std::make_tuple(384, 56, 832)));
  data.push_back(std::move(std::make_tuple(192, 56, 832)));
  data.push_back(std::move(std::make_tuple(384, 56, 1728)));
  data.push_back(std::move(std::make_tuple(48, 56, 832)));
  data.push_back(std::move(std::make_tuple(128, 56, 1200)));
  data.push_back(std::move(std::make_tuple(128, 56, 832)));

  for (auto it = data.begin(); it < data.end(); ++it) {
    size_t m = std::get<0>(*it);
    size_t n = std::get<1>(*it);
    size_t k = std::get<2>(*it);
    std::vector<int8_t> a(m * k);
    std::vector<float> a_ref(m * k);
    for (size_t i = 0; i < m * k; ++i) {
      a[i] = static_cast<int8_t>(threshold * static_cast<float>(std::rand()) / RAND_MAX);
      a_ref[i] = static_cast<float>(a[i]);
    }
    std::vector<uint8_t> b(n * k);
    std::vector<float> b_ref(n * k);
    for (size_t i = 0; i < n * k; ++i) {
      b[i] = static_cast<uint8_t>(threshold * static_cast<float>(std::rand()) / RAND_MAX);
      b_ref[i] = static_cast<float>(b[i]);
    }
    std::vector<int> c(m * n);
    std::vector<float> c_ref(m * n);
    MixPrecisionGemm(static_cast<ORDER>(101), static_cast<TRANSPOSE>(111), static_cast<TRANSPOSE>(112), m, n, k,
                     a.data(), k, b.data(), k, c.data(), n, 0.5);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a_ref.data(), k, b_ref.data(), k, 0.0f,
                c_ref.data(), n);
    for (int i = 0; i < c.size(); ++i) {
      DOUBLES_EQUAL(c_ref[i], static_cast<float>(c[i]), 1e-10);
    }
  }
}

int main(int argc, char** argv) {
  return RUN_ALL_TESTS(argc, argv);
}
