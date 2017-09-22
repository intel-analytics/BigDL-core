#ifndef TENSOR_H
#define TENSOR_H

#include "alloc.h"

struct Shape {
  size_t dim_;
  std::vector<size_t> shape_;

  Shape() : dim_(0), shape_(0, 0) {
  }

  Shape(size_t dim) : dim_(dim), shape_(dim, 0) {
  }

  Shape(size_t dim, size_t *shape) : dim_(dim), shape_(dim) {
    shape_.assign(shape, shape + dim);
  }

  Shape(Shape *p) : dim_(p->dim_), shape_(p->shape_) {
  }

  size_t Count() {
    if (dim_ == 0) {
      return 0;
    } else {
      size_t count = 1;
      for (auto it = shape_.begin(); it < shape_.begin() + dim_; ++it) {
        count *= *it;
      }
      return count;
    }
  }

  size_t operator[](size_t index) {
    return shape_[index];
  }
};

Shape make_shape(size_t x) {
  size_t shape[1] = {x};
  return Shape(1, shape);
}

Shape make_shape(size_t x, size_t y) {
  size_t shape[2] = {x, y};
  return Shape(2, shape);
}

Shape make_shape(size_t x, size_t y, size_t z) {
  size_t shape[3] = {x, y, z};
  return Shape(3, shape);
}

Shape make_shape(size_t x, size_t y, size_t z, size_t w) {
  size_t shape[4] = {x, y, z, w};
  return Shape(4, shape);
}

template <typename DType>
struct Tensor {
  Shape shape_;
  DType *data_;
  bool data_owner_;

  Tensor(Shape s) : shape_(s), data_(NULL), data_owner_(false) {
  }

  Tensor(Shape s, size_t alignment) : shape_(s), data_(NULL), data_owner_(true) {
    Allocate(alignment);
  }

  Tensor(Shape s, DType *data) : shape_(s), data_(data), data_owner_(false) {
  }

  ~Tensor() {
    if (data_ && data_owner_) {
      aligned_free(data_);
    }
  }

  size_t Count() {
    return shape_.Count();
  }

  size_t Size() {
    return sizeof(DType) * shape_.Count();
  }

  size_t ExclusiveSize() {
    return (data_owner_ == true) ? Size() : 0;
  }

  void Allocate(size_t alignment = 64) {
    data_owner_ = true;
    aligned_malloc(reinterpret_cast<void **>(&data_), alignment, Size());
  }

  void SetData(DType *data) {
    data_owner_ = false;
    data_ = data;
  }

  size_t operator[](size_t index) {
    return data_[index];
  }
};

template <typename SrcType, typename DstType>
struct QuantizedTensor : public Tensor<DstType> {
  Tensor<SrcType> min_;
  Tensor<SrcType> max_;
  Tensor<SrcType> ratio_;
  // why do we need realshape when we already have shape?
  // Because quantizedtensor may be used for computation and pad and alignment, this may cause real shape is different
  // from ori_shape
  Shape ori_shape_;

  QuantizedTensor(Shape quantized_shape, Shape meta_shape, size_t alignment)
      : Tensor<DstType>(quantized_shape, alignment),
        min_(meta_shape, alignment),
        max_(meta_shape, alignment),
        ratio_(meta_shape, alignment),
        ori_shape_() {
  }

  QuantizedTensor(Shape quantized_shape, Shape meta_shape, Shape ori_shape, size_t alignment)
      : Tensor<DstType>(quantized_shape, alignment),
        ori_shape_(ori_shape),
        min_(meta_shape, alignment),
        max_(meta_shape, alignment),
        ratio_(meta_shape, alignment) {
  }

  QuantizedTensor(Shape quantized_shape, Shape meta_shape, Shape ori_shape)
      : Tensor<DstType>(quantized_shape),
        ori_shape_(ori_shape),
        min_(meta_shape),
        max_(meta_shape),
        ratio_(meta_shape) {
  }

  size_t Size() {
    return sizeof(SrcType) * this->shape_.Count() + min_.Size() + max_.Size() + ratio_.Size();
  }
};

//}
#endif
