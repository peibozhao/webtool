#pragma once

#include <nvjpeg.h>
#include <format>
#include <NvInfer.h>

__global__ void convertUint8ToFloat(const uint8_t *in, int size, float *out);

__global__ void convertFloatToUint8(const float *in, int size, uint8_t *out);

void launchConvertUint8ToFloat(const uint8_t *in, int size, float *out,
                               cudaStream_t stream);

void launchConvertFloatToUint8(const float *in, int size, uint8_t *out,
                               cudaStream_t stream);
template <>
struct std::formatter<nvinfer1::Dims> {
  constexpr auto parse(format_parse_context &fpc) {
    return fpc.begin();
  }

  auto format(const nvinfer1::Dims &dims,
              format_context &fc) const {
    auto out = std::format_to(fc.out(), "{{");
    for (int idx = 0; idx < dims.nbDims; ++idx) {
      out = std::format_to(out, "{}", dims.d[idx]);
      if (idx == dims.nbDims - 1) {
        continue;
      }
      out = std::format_to(out, ",");
    }
    out = std::format_to(out, "}}");
    return out;
  }
};
