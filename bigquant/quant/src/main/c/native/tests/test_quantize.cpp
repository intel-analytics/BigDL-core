#include <iostream>
#include <array>
#include <tuple>
#include <vector>
#include <algorithm>
#include "../base.h"
#include "../common.h"
#include "../ops/ops.h"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"

TEST_GROUP(Quantize) {

};

TEST(Quantize, QuantizeFloat2Int8) {
  std::vector<size_t> length = {4, 8, 16, 32, 64, 128, 256, 512, 1023, 1024, 1025, 1026};
  std::vector<float> thresholds = {32.0f, 63.0f, 64.0f, 127.0f};
  for (auto it_len = length.begin(); it_len < length.end(); ++it_len) {
    for (auto it_threshold = thresholds.begin(); it_threshold < thresholds.end(); ++it_threshold) {
      std::vector<float> src(*it_len);
      std::vector<int8_t> dst(*it_len);
      std::generate(src.begin(), src.end(), []{return static_cast<float>(std::rand()) / RAND_MAX;});
      float max, min, ratio;
      PadQuantize<float>(dst.data(), src.size(), src.size(), src.data(), min, max, ratio, *it_threshold);
      float max_ref, min_ref, ratio_ref;
      auto minmax = std::minmax_element(std::begin(src), std::end(src));
      min_ref = *(minmax.first);
      max_ref = *(minmax.second);
      ratio_ref = (std::abs(max) > std::abs(min))? (*it_threshold / std::abs(max)) : (*it_threshold / std::abs(min));
      DOUBLES_EQUAL(max_ref, max, 1e-6);
      DOUBLES_EQUAL(min_ref, min, 1e-6);
      DOUBLES_EQUAL(ratio_ref, ratio, 1e-6);
      for (size_t i = 0; i < src.size(); ++i) {
        DOUBLES_EQUAL(std::round(src.data()[i] * ratio_ref), static_cast<float>(dst.data()[i]), 1e-6);
      }
    }
  }
}

TEST(Quantize, QuantizeDouble2Int8) {
  std::vector<size_t> length = {4, 8, 16, 32, 64, 128, 256, 512, 1023, 1024, 1025, 1026};
  std::vector<float> thresholds = {32.0f, 63.0f, 64.0f, 127.0f};
  for (auto it_len = length.begin(); it_len < length.end(); ++it_len) {
    for (auto it_threshold = thresholds.begin(); it_threshold < thresholds.end(); ++it_threshold) {
      std::vector<double> src(*it_len);
      std::vector<int8_t> dst(*it_len);
      std::generate(src.begin(), src.end(), []{return static_cast<double>(std::rand()) / RAND_MAX;});
      double max, min, ratio;
      PadQuantize<double>(dst.data(), src.size(), src.size(), src.data(), min, max, ratio, *it_threshold);
      double max_ref, min_ref, ratio_ref;
      auto minmax = std::minmax_element(std::begin(src), std::end(src));
      min_ref = *(minmax.first);
      max_ref = *(minmax.second);
      ratio_ref = (std::abs(max) > std::abs(min))? (*it_threshold / std::abs(max)) : (*it_threshold / std::abs(min));
      DOUBLES_EQUAL(max_ref, max, 1e-6);
      DOUBLES_EQUAL(min_ref, min, 1e-6);
      DOUBLES_EQUAL(ratio_ref, ratio, 1e-6);
      for (size_t i = 0; i < src.size(); ++i) {
        DOUBLES_EQUAL(std::round(src.data()[i] * ratio_ref), static_cast<double>(dst.data()[i]), 1e-6);
      }
    }
  }
}

TEST(Quantize, QuantizeFloat2UInt8) {
  std::vector<size_t> length = {4, 8, 16, 32, 64, 128, 256, 512, 1023, 1024, 1025, 1026};
  std::vector<float> thresholds = {32.0f, 63.0f, 64.0f, 127.0f, 255.0f};
  for (auto it_len = length.begin(); it_len < length.end(); ++it_len) {
    for (auto it_threshold = thresholds.begin(); it_threshold < thresholds.end(); ++it_threshold) {
      std::vector<float> src(*it_len);
      std::vector<uint8_t> dst(*it_len);
      std::generate(src.begin(), src.end(), []{return static_cast<float>(std::rand()) / RAND_MAX;});
      float max, min, ratio;
      PadQuantize<float>(dst.data(), src.size(), src.size(), src.data(), min, max, ratio, *it_threshold);
      float max_ref, min_ref, ratio_ref;
      auto minmax = std::minmax_element(std::begin(src), std::end(src));
      min_ref = *(minmax.first);
      max_ref = *(minmax.second);
      ratio_ref = (*it_threshold) / (max_ref - min_ref);
      DOUBLES_EQUAL(max_ref, max, 1e-6);
      DOUBLES_EQUAL(min_ref, min, 1e-6);
      DOUBLES_EQUAL(ratio_ref, ratio, 1e-6);
      for (size_t i = 0; i < src.size(); ++i) {
        DOUBLES_EQUAL(std::round((src.data()[i] - min_ref) * ratio_ref), static_cast<float>(dst.data()[i]), 1e-6);
      }
    }
  }
}

TEST(Quantize, QuantizeDouble2UInt8) {
  std::vector<size_t> length = {4, 8, 16, 32, 64, 128, 256, 512, 1023, 1024, 1025, 1026};
  std::vector<float> thresholds = {32.0f, 63.0f, 64.0f, 127.0f, 255.0f};
  for (auto it_len = length.begin(); it_len < length.end(); ++it_len) {
    for (auto it_threshold = thresholds.begin(); it_threshold < thresholds.end(); ++it_threshold) {
      std::vector<double> src(*it_len);
      std::vector<uint8_t> dst(*it_len);
      std::generate(src.begin(), src.end(), []{return static_cast<double>(std::rand()) / RAND_MAX;});
      double max, min, ratio;
      PadQuantize<double>(dst.data(), src.size(), src.size(), src.data(), min, max, ratio, *it_threshold);
      double max_ref, min_ref, ratio_ref;
      auto minmax = std::minmax_element(std::begin(src), std::end(src));
      min_ref = *(minmax.first);
      max_ref = *(minmax.second);
      ratio_ref = (*it_threshold) / (max_ref - min_ref);
      DOUBLES_EQUAL(max_ref, max, 1e-6);
      DOUBLES_EQUAL(min_ref, min, 1e-6);
      DOUBLES_EQUAL(ratio_ref, ratio, 1e-6);
      for (size_t i = 0; i < src.size(); ++i) {
        DOUBLES_EQUAL(std::round((src.data()[i] - min_ref) * ratio_ref), static_cast<double>(dst.data()[i]), 1e-6);
      }
    }
  }
}

TEST(Quantize, PADQuantizeShuffle2DUInt8) {
  const size_t m_block = CONV_SHUFFLE_KERNEL_M;
  const size_t n_block = CONV_SHUFFLE_KERNEL_K;
  std::vector<std::pair<size_t, size_t>> problemset;
  problemset.push_back(std::make_pair(8, 8));
  problemset.push_back(std::make_pair(32, 32));
  problemset.push_back(std::make_pair(64, 64));
  problemset.push_back(std::make_pair(128, 128));
  problemset.push_back(std::make_pair(128, 333));
  std::vector<float> thresholds = {32.0f, 64.0f, 127.0f};
  for (auto problem_it = problemset.begin(); problem_it < problemset.end(); ++problem_it) {
    for (auto thres_it = thresholds.begin(); thres_it < thresholds.end(); ++thres_it) {
      size_t m = problem_it->first;
      size_t n = problem_it->second;
      std::vector<float> src(m * n);
      std::generate(src.begin(), src.end(), []{return static_cast<float>(std::rand()) / RAND_MAX;});
      std::vector<float> min_ref(m);
      std::vector<float> max_ref(m);
      std::vector<float> ratio_ref(m);
      auto src_iter = src.begin();
      for (size_t y = 0; y < m; ++y) {
        size_t begin = y * n;
        size_t end = (y + 1) * n;
        auto minmax = std::minmax_element(src_iter + begin, src_iter + end);
        min_ref[y] = *(minmax.first);
        max_ref[y] = *(minmax.second);
        ratio_ref[y] = *thres_it / (max_ref[y] - min_ref[y]);
      }
      size_t pad_m = GetAlignmentLength(m, m_block);
      size_t pad_n = GetAlignmentLength(n, n_block);
      std::vector<uint8_t> dst(pad_m * pad_n);
      std::vector<float> min(m);
      std::vector<float> max(m);
      std::vector<float> ratio(m);
      shuffle::PadQuantizeShuffle2D<float, m_block, n_block>(dst.data(), m, n, pad_m, pad_n, src.data(), min.data(), max.data(), ratio.data(), *thres_it);
      for (size_t i = 0; i < m; ++i) {
        DOUBLES_EQUAL(min_ref[i], min[i], 1e-6);
        DOUBLES_EQUAL(max_ref[i], max[i], 1e-6);
        DOUBLES_EQUAL(1.0 / ratio_ref[i], ratio[i], 1e-6);
      }
      size_t x_block_num = static_cast<size_t>(std::ceil(1.0 * n / n_block)); // x_block_num
      size_t block_size = m_block * n_block;
      for (size_t i = 0; i < m; ++i) {
        size_t y_block_id = i / m_block;
        for (size_t j = 0; j < n; ++j) {
          size_t x_block_id = j / n_block;
          size_t block_id = y_block_id * x_block_num + x_block_id;
          BYTES_EQUAL(*(dst.data() + block_id * block_size + (i % m_block) * n_block + (j % n_block)), static_cast<uint8_t>(std::round((*(src.data() + i * n + j) - min[i]) * ratio_ref[i])));
        }
      }
    }
  }
}


TEST(Quantize, PADQuantizeShuffle2DInt8) {
  const size_t m_block = CONV_SHUFFLE_KERNEL_M;
  const size_t n_block = CONV_SHUFFLE_KERNEL_K;
  std::vector<std::pair<size_t, size_t>> problemset;
  problemset.push_back(std::move(std::make_pair(8, 8)));
  problemset.push_back(std::move(std::make_pair(32, 32)));
  problemset.push_back(std::move(std::make_pair(64, 64)));
  problemset.push_back(std::move(std::make_pair(128, 128)));
  problemset.push_back(std::move(std::make_pair(128, 333)));
  std::vector<float> thresholds = {32.0f, 64.0f, 127.0f};
  for (auto problem_it = problemset.begin(); problem_it < problemset.end(); ++problem_it) {
    for (auto thres_it = thresholds.begin(); thres_it < thresholds.end(); ++thres_it) {
      size_t m = problem_it->first;
      size_t n = problem_it->second;
      std::vector<float> src(m * n);
      std::generate(src.begin(), src.end(), []{return static_cast<float>(std::rand()) / RAND_MAX;});
      std::vector<float> min_ref(m);
      std::vector<float> max_ref(m);
      std::vector<float> ratio_ref(m);
      auto src_iter = src.begin();
      for (size_t y = 0; y < m; ++y) {
        size_t begin = y * n;
        size_t end = (y + 1) * n;
        auto minmax = std::minmax_element(src_iter + begin, src_iter + end);
        min_ref[y] = *(minmax.first);
        max_ref[y] = *(minmax.second);
        ratio_ref[y] = *thres_it / std::max(std::abs(max_ref[y]), std::abs(min_ref[y]));
      }
      size_t pad_m = GetAlignmentLength(m, m_block);
      size_t pad_n = GetAlignmentLength(n, n_block);
      std::vector<int8_t> dst(pad_m * pad_n);
      std::vector<float> min(m);
      std::vector<float> max(m);
      std::vector<float> ratio(m);
      shuffle::PadQuantizeShuffle2D<float, m_block, n_block>(dst.data(), m, n, pad_m, pad_n, src.data(), min.data(), max.data(), ratio.data(), *thres_it);
      for (size_t i = 0; i < m; ++i) {
        DOUBLES_EQUAL(min_ref[i], min[i], 1e-6);
        DOUBLES_EQUAL(max_ref[i], max[i], 1e-6);
        DOUBLES_EQUAL(1.0 / ratio_ref[i], ratio[i], 1e-6);
      }
      size_t x_block_num = static_cast<size_t>(std::ceil(1.0 * n / n_block)); // x_block_num
      size_t block_size = m_block * n_block;
      for (size_t i = 0; i < m; ++i) {
        size_t y_block_id = i / m_block;
        for (size_t j = 0; j < n; ++j) {
          size_t x_block_id = j / n_block;
          size_t block_id = y_block_id * x_block_num + x_block_id;
          BYTES_EQUAL(*(dst.data() + block_id * block_size + (i % m_block) * n_block + (j % n_block)), static_cast<uint8_t>(std::round(*(src.data() + i * n + j) * ratio_ref[i])));
        }
      }
    }
  }
}

TEST_GROUP(Im2Col) {

};

TEST(Im2Col, PADQuantizeShuffleNHWCIM2COL) {
  const size_t m_block = CONV_SHUFFLE_KERNEL_N;
  const size_t n_block = CONV_SHUFFLE_KERNEL_K;
  std::vector<float> thresholds = {32.0f, 64.0f, 127.0f};
  // tuple order: batch_size, height_in, width_in, kernel_size, channel_in, group, stride, pad
  std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t>> data;
  data.push_back(std::move(std::make_tuple(32, 37, 37, 5, 128, 1, 2, 2)));
  data.push_back(std::move(std::make_tuple(32, 15, 15, 5, 256, 1, 2, 2)));
  data.push_back(std::move(std::make_tuple(32, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(127, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 0)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 255, 1, 1, 0)));
  for (auto thres_it = thresholds.begin(); thres_it < thresholds.end(); ++thres_it) {
    for (auto it = data.begin(); it < data.end(); ++it) {
      size_t batch_size = std::get<0>(*it);
      size_t height_in = std::get<1>(*it);
      size_t width_in = std::get<2>(*it);
      size_t kernel = std::get<3>(*it);
      size_t channel = std::get<4>(*it);
      size_t group = std::get<5>(*it);
      size_t stride = std::get<6>(*it);
      size_t pad = std::get<7>(*it);
      size_t height_out = GetConvOutSize(height_in, kernel, stride, pad, 1);
      size_t width_out = GetConvOutSize(width_in, kernel, stride, pad, 1);
      size_t m = batch_size * height_out * width_out;
      size_t k = kernel * kernel * channel;
      size_t m_ = GetAlignmentLength(batch_size * height_out * width_out, m_block);
      size_t k_ = GetAlignmentLength(kernel * kernel * channel, n_block);
      std::vector<float> src(batch_size * channel * height_in * width_in);
      std::generate(src.begin(), src.end(), []{return static_cast<float>(std::rand()) / RAND_MAX;});

      std::vector<uint8_t> dst_ref(m_ * k_);
      std::vector<float> min_ref(m);
      std::vector<float> max_ref(m);
      std::vector<float> ratio_ref(m);

      std::vector<uint8_t> dst(m_ * k_);
      std::vector<float> min(m);
      std::vector<float> max(m);
      std::vector<float> ratio(m);

      uint8_t* dst_array[1];
      dst_array[0] = dst.data();
      float* min_array[1];
      min_array[0] = min.data();
      float* max_arrary[1];
      max_arrary[0] = max.data();
      float* ratio_array[1];
      ratio_array[0] = ratio.data();

      shuffle::PadQuantizeShuffleIm2colWrapper<float, NHWC>(src.data(), batch_size, channel, group, height_in, width_in, \
          kernel, kernel, pad, pad, stride, stride, 1, 1, dst_array, min_array, max_arrary, ratio_array, NULL, 127.0f);

      shuffle::PadQuantizeShuffleIm2colRef<float, m_block, n_block, NHWC>(src.data(), batch_size, channel, height_in, width_in, \
          kernel, kernel, pad, pad, stride, stride, 1, 1, dst_ref.data(), min_ref.data(), max_ref.data(), ratio_ref.data(), 127.0f);

      for (size_t i = 0; i < min.size(); ++i) {
        DOUBLES_EQUAL(max_ref[i], max[i], 1e-6);
        DOUBLES_EQUAL(min_ref[i], min[i], 1e-6);
        DOUBLES_EQUAL(ratio_ref[i], ratio[i], 1e-6);
      }

      for (size_t j = 0; j < dst_ref.size(); ++j) {
        //DOUBLES_EQUAL(static_cast<float>(dst_ref[j]), static_cast<float>(dst[j]), 1.0f);
        BYTES_EQUAL(dst_ref[j], dst[j]);
      }
    }
  }
}


TEST(Im2Col, PADQuantizeShuffleNCHWIM2COL) {
  const size_t m_block = CONV_SHUFFLE_KERNEL_N;
  const size_t n_block = CONV_SHUFFLE_KERNEL_K;
  std::vector<float> thresholds = {32.0f, 64.0f, 127.0f};
  // tuple order: batch_size, height_in, width_in, kernel_size, channel_in, group, stride, pad
  std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t>> data;
  data.push_back(std::move(std::make_tuple(32, 37, 37, 5, 128, 1, 2, 2)));
  data.push_back(std::move(std::make_tuple(32, 15, 15, 5, 256, 1, 2, 2)));
  data.push_back(std::move(std::make_tuple(32, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(127, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 0)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 255, 1, 1, 0)));
  for (auto thres_it = thresholds.begin(); thres_it < thresholds.end(); ++thres_it) {
    for (auto it = data.begin(); it < data.end(); ++it) {
      size_t batch_size = std::get<0>(*it);
      size_t height_in = std::get<1>(*it);
      size_t width_in = std::get<2>(*it);
      size_t kernel = std::get<3>(*it);
      size_t channel = std::get<4>(*it);
      size_t group = std::get<5>(*it);
      size_t stride = std::get<6>(*it);
      size_t pad = std::get<7>(*it);
      size_t height_out = GetConvOutSize(height_in, kernel, stride, pad, 1);
      size_t width_out = GetConvOutSize(width_in, kernel, stride, pad, 1);
      size_t m = batch_size * height_out * width_out;
      size_t k = kernel * kernel * channel;
      size_t m_ = GetAlignmentLength(batch_size * height_out * width_out, m_block);
      size_t k_ = GetAlignmentLength(kernel * kernel * channel, n_block);
      std::vector<float> src(batch_size * channel * height_in * width_in);
      std::generate(src.begin(), src.end(), []{return static_cast<float>(std::rand()) / RAND_MAX;});

      std::vector<uint8_t> dst_ref(m_ * k_);
      std::vector<float> min_ref(m);
      std::vector<float> max_ref(m);
      std::vector<float> ratio_ref(m);

      std::vector<uint8_t> dst(m_ * k_);
      std::vector<float> min(m);
      std::vector<float> max(m);
      std::vector<float> ratio(m);

      uint8_t* dst_array[1];
      dst_array[0] = dst.data();
      float* min_array[1];
      min_array[0] = min.data();
      float* max_arrary[1];
      max_arrary[0] = max.data();
      float* ratio_array[1];
      ratio_array[0] = ratio.data();

      shuffle::PadQuantizeShuffleIm2colWrapper<float, NCHW>(src.data(), batch_size, channel, group, height_in, width_in, \
          kernel, kernel, pad, pad, stride, stride, 1, 1, dst_array, min_array, max_arrary, ratio_array, NULL, *thres_it);

      shuffle::PadQuantizeShuffleIm2colRef<float, m_block, n_block, NCHW>(src.data(), batch_size, channel, height_in, width_in, \
          kernel, kernel, pad, pad, stride, stride, 1, 1, dst_ref.data(), min_ref.data(), max_ref.data(), ratio_ref.data(), *thres_it);

      for (size_t i = 0; i < min.size(); ++i) {
        DOUBLES_EQUAL(max_ref[i], max[i], 1e-6);
        DOUBLES_EQUAL(min_ref[i], min[i], 1e-6);
        DOUBLES_EQUAL(ratio_ref[i], ratio[i], 1e-6);
      }
      for (size_t j = 0; j < dst_ref.size(); ++j) {
        BYTES_EQUAL(dst_ref[j], dst[j]);
      }
    }
  }
}

/*
#if defined(AVX512)
TEST(Im2Col, PADQuantizeNHWCIM2COL) {
  const size_t m_block = CONV_KERNEL_N;
  const size_t n_block = CONV_KERNEL_K;
  std::vector<float> thresholds = {32.0f, 64.0f, 127.0f};
  // tuple order: batch_size, height_in, width_in, kernel_size, channel_in, group, stride, pad
  std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t>> data;
  data.push_back(std::move(std::make_tuple(32, 37, 37, 5, 128, 1, 2, 2)));
  data.push_back(std::move(std::make_tuple(32, 15, 15, 5, 256, 1, 2, 2)));
  data.push_back(std::move(std::make_tuple(32, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(127, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 1)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 256, 1, 1, 0)));
  data.push_back(std::move(std::make_tuple(128, 15, 15, 3, 255, 1, 1, 0)));
  for (auto thres_it = thresholds.begin(); thres_it < thresholds.end(); ++thres_it) {
    for (auto it = data.begin(); it < data.end(); ++it) {
      size_t batch_size = std::get<0>(*it);
      size_t height_in = std::get<1>(*it);
      size_t width_in = std::get<2>(*it);
      size_t kernel = std::get<3>(*it);
      size_t channel = std::get<4>(*it);
      size_t group = std::get<5>(*it);
      size_t stride = std::get<6>(*it);
      size_t pad = std::get<7>(*it);
      size_t height_out = GetConvOutSize(height_in, kernel, stride, pad, 1);
      size_t width_out = GetConvOutSize(width_in, kernel, stride, pad, 1);
      size_t m = batch_size * height_out * width_out;
      size_t k = kernel * kernel * channel;
      size_t m_ = GetAlignmentLength(batch_size * height_out * width_out, m_block);
      size_t k_ = GetAlignmentLength(kernel * kernel * channel, n_block);
      std::vector<float> src(batch_size * channel * height_in * width_in);
      std::generate(src.begin(), src.end(), []{return static_cast<float>(std::rand()) / RAND_MAX;});

      std::vector<uint8_t> dst_ref(m * k);
      std::vector<float> min_ref(m);
      std::vector<float> max_ref(m);
      std::vector<float> ratio_ref(m);

      std::vector<uint8_t> dst(m_ * k_);
      std::vector<float> min(m);
      std::vector<float> max(m);
      std::vector<float> ratio(m);

      uint8_t* dst_array[1];
      dst_array[0] = dst.data();
      float* min_array[1];
      min_array[0] = min.data();
      float* max_arrary[1];
      max_arrary[0] = max.data();
      float* ratio_array[1];
      ratio_array[0] = ratio.data();

      PadQuantizeIm2colWrapper<float, NHWC>(src.data(), batch_size, channel, group, height_in, width_in, \
          kernel, kernel, pad, pad, stride, stride, 1, 1, dst_array, min_array, max_arrary, ratio_array, NULL, *thres_it);

      QuantizeIm2colRef<float, NHWC>(src.data(), batch_size, channel, height_in, width_in, \
          kernel, kernel, pad, pad, stride, stride, 1, 1, dst_ref.data(), min_ref.data(), max_ref.data(), ratio_ref.data(), *thres_it);


      for (size_t i = 0; i < min.size(); ++i) {
        DOUBLES_EQUAL(max_ref[i], max[i], 1e-6);
        DOUBLES_EQUAL(min_ref[i], min[i], 1e-6);
        DOUBLES_EQUAL(ratio_ref[i], ratio[i], 1e-6);
      }
      for (size_t y = 0; y < m_; ++y) {
        for (size_t x = 0; x < k_; ++x) {
          if ((y < m) && (x < k)) {
            BYTES_EQUAL(dst_ref[y * k + x], dst[y * k_ + x]);
            // DOUBLES_EQUAL(static_cast<float>(dst_ref[y * k + x]), static_cast<float>(dst[y * k_ + x]), 1.1f);
          }
        }
      }
    }
  }
}
#endif
*/

int main(int argc, char** argv) {
  return RUN_ALL_TESTS(argc, argv);
}
