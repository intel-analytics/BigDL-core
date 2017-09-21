#ifndef OPS_DOT_H
#define OPS_DOT_H

#include "../base.h"
#include "../common.h"
#include "./kernel/streamdot.h"
#include "./kernel/multistream4x2_dot.h"
namespace dot {

void Dot(int8_t* pa, uint8_t* pb, int& result, size_t length) {
  kernel::dot::ApplyKernel(pa, pb, result, length);
}

void Dot(int8_t* pa, uint8_t* pb, float& result, size_t length, float ratio_a, float a_sum, float ratio_b,
         float min_b) {
  kernel::dot::ApplyKernel(pa, pb, result, length, ratio_a, a_sum, ratio_b, min_b);
}
}

#endif
