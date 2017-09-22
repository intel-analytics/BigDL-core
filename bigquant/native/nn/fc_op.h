#ifndef NN_FC_OP_H
#define NN_FC_OP_H

#include "base_fc.h"
#include "shuffle_fc.h"

struct FCOp {
  FCOp() : algo_id_(AUTO_SELECT_FC), algo_(NULL) {
  }

  ~FCOp() {
    delete algo_;
  }

  void SetupFCKernelParameter(LAYOUT layout, size_t channel_out, size_t channel_in, FC_ALGORITHM algo) {
    fc_kernel_desc_ = {layout, channel_out, channel_in};
    ChooseAlgo(algo);
  }

  void SetupFCDataParameter(size_t batch_size, size_t channel_in) {
    fc_data_desc_ = {batch_size, channel_in};
  }

  void ChooseAlgo(FC_ALGORITHM algo_id) {
    algo_id_ = algo_id;
    switch (algo_id_) {
      case SHUFFLE_FC: {
        algo_ = new ShuffleFCAlgo();
        break;
      }
      default: {
        algo_ = new ShuffleFCAlgo();
        break;
      }
    }
  }

  void InitWeight(float *weight) {
    algo_->InitWeight(weight, fc_kernel_desc_);
  }

  void Execute(float *out, float *data, float *bias, size_t batch_size, size_t channel_in) {
    SetupFCDataParameter(batch_size, channel_in);
    algo_->Execute(out, data, bias, fc_data_desc_, fc_kernel_desc_);
  }

  FC_ALGORITHM algo_id_;
  BaseFCAlgo *algo_;
  FCKernelDesc fc_kernel_desc_;
  FCDataDesc fc_data_desc_;
};

#endif
