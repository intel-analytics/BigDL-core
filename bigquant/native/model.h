#ifndef MODEL_H
#define MODEL_H

void DequantizeModel(float *dst, int8_t *src, float *src_min, float *src_max, size_t c_out, size_t c_in,
                     size_t kernel_h, size_t kernel_w) {
  for (size_t c_o = 0; c_o < c_out; ++c_o) {
    size_t meta_index = c_o;
    for (size_t k = 0; k < c_in * kernel_h * kernel_w; ++k) {
      size_t index = c_o * c_in * kernel_h * kernel_w + k;
      dst[index] =
          1.0 * static_cast<float>(src[index]) / 127.0 * fmaxf(fabs(src_max[meta_index]), fabs(src_min[meta_index]));
    }
  }
}
#endif
