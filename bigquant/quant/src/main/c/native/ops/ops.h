#ifndef OPS_OPS_H
#define OPS_OPS_H



template <typename DType>
void FindMinMaxValue(const DType *p, size_t length, DType &min, DType &max);

template <typename DType>
void OMPFindMinMaxValue(DType *p, size_t length, DType &min_value, DType &max_value, float threshold);

template <typename SrcType>
void PadQuantize(int8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max, SrcType &ratio, float threshold);

template <typename SrcType>
void PadQuantize(uint8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max, SrcType &ratio, float threshold);

template <typename SrcType>
void ParallelPadQuantize(int8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max, SrcType &ratio, float threshold);

template <typename SrcType>
void ParallelPadQuantize(uint8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max, SrcType &ratio, float threshold);

template <typename SrcType>
void Quantize2D(int8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, SrcType *src, SrcType *min, SrcType *max, SrcType *ratio);

template <typename SrcType>
void Quantize2D(uint8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, SrcType *src, SrcType *min, SrcType *max, SrcType *ratio);

template <typename DType, LAYOUT layout>
void UnGroupKernel(DType* dst[], DType *src, size_t group, size_t channel_out, size_t channel_in, size_t hxw);

template <typename DType>
void Transpose(DType *dst, DType *src, size_t m, size_t n);

template <typename DType>
void TransformLayout(LAYOUT dst_layout, LAYOUT src_layout, DType *dst, DType *src, size_t batch_size, size_t channels, size_t hxw);

typedef enum ORDER {RowMajor=101, ColMajor=102} ORDER;
typedef enum TRANSPOSE {NoTrans=111, Trans=112} TRANSPOSE;

template <typename DType, LAYOUT layout>
void PadQuantizeIm2colWrapper(DType* data,
            size_t batch_size,
            size_t channels_per_group, size_t groups,
            size_t height, size_t width,
            size_t kernel_h, size_t kernel_w,
            size_t pad_h, size_t pad_w,
            size_t stride_h, size_t stride_w,
            size_t dilation_h, size_t dilation_w,
            uint8_t *data_col[],
            DType *min[],
            DType *max[],
            DType *ratio[],
            DType *workspace,
            float sw_threshold = 255.0f,
            bool transpose = false);

void MixPrecisionGemm(ORDER order,
                  enum TRANSPOSE transA, enum TRANSPOSE transB,
                  int m, int n, int k,
                  int8_t *a, int lda, uint8_t *b, int ldb, int *c, int ldc, float fault_tolerance);

namespace shuffle {

template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadShuffle2D(DType *dst, size_t m, size_t n, DType *src);

template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadQuantizeShuffle(int8_t *dst, size_t m, size_t n, DType *src, DType &min, DType &max, DType &ratio, float sw_threshold);

template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadQuantizeShuffle2D(int8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, DType *src, DType *min, DType *max, DType *ratio, float sw_threshold);

template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadQuantizeShuffle2D(uint8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, DType *src, DType *min, DType *max, DType *ratio, float sw_threshold);

template <typename DType, LAYOUT layout>
void PadQuantizeShuffleIm2colWrapper(DType* data,
            size_t batch_size,
            size_t channels_per_group, size_t groups,
            size_t height, size_t width,
            size_t kernel_h, size_t kernel_w,
            size_t pad_h, size_t pad_w,
            size_t stride_h, size_t stride_w,
            size_t dilation_h, size_t dilation_w,
            uint8_t *data_col[],
            DType *min[],
            DType *max[],
            DType *ratio[],
            DType *workspace,
            float sw_threshold = 255.0f,
            bool transpose = false);

template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
void ConvShuffleGEMM(int8_t* pa, uint8_t* pb, float* pc, size_t m, size_t n, size_t k,
  float* ratio_a, float* ratio_b, float* kernel_sum, float* min_b, float* bias,
  size_t batch_size, size_t groups, size_t channel_per_group, size_t cur_group, size_t height_out, size_t width_out,
  float fault_tolerance = 0.5, size_t pad_m = 0, size_t pad_n = 0,
  bool conv_relu_fusion = false, bool conv_bn_fusion = false, bool conv_bn_relu_fusion = false, bool conv_relu_bn_fusion = false,
  float *global_mean = NULL, float *mul_variance_coeff = NULL, float *scale = NULL, float *shift = NULL);

}


#include "find_extreme.h"
#include "quantize.h"
#include "group.h"
#include "layout.h"
#include "./shuffle/pad_shuffle.h"
#include "./shuffle/shuffle_im2col.h"
#include "./shuffle/shuffle_igemm.h"
#include "./mixprecison_gemm.h"
#endif
