
#include "cuda_utils.h"

__global__ void convertUint8ToFloat(const uint8_t *in, int size, float *out) {
  int x =  blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= size)
    return;
  out[x] = in[x] / 255.;
}

__global__ void convertFloatToUint8(const float *in, int size, uint8_t *out) {
  int x =  blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= size)
    return;
  out[x] = in[x] * 255;
}

void launchConvertUint8ToFloat(const uint8_t *in, int size, float *out,
                               cudaStream_t stream) {
  convertUint8ToFloat<<<size / 64, 64, 0, stream>>>(in, size, out);
}

void launchConvertFloatToUint8(const float *in, int size, uint8_t *out,
                               cudaStream_t stream) {
  convertFloatToUint8<<<size / 64, 64, 0, stream>>>(in, size, out);
}
