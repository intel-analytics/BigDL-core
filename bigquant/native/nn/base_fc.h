#ifndef NN_BASE_FC_H
#define NN_BASE_FC_H

#include "../base.h"
#include "../common.h"
#include "../tensor.h"
#include "../ops/ops.h"

struct FCKernelDesc {
  LAYOUT layout_;
  size_t channel_out_;
  size_t channel_in_;
};
struct FCDataDesc {
  size_t batch_size_;
  size_t channel_in_;
};

struct BaseFCAlgo {
  BaseFCAlgo() {
  }
  virtual ~BaseFCAlgo() {
  }
  virtual void InitWeight(float *weight, FCKernelDesc &fc_kernel_desc) = 0;
  virtual void Execute(float *out, float *data, float *bias, FCDataDesc &fc_data_desc,
                       FCKernelDesc &fc_kernel_desc) = 0;
};

#endif
