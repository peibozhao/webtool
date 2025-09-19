#pragma once

#include <nvjpeg.h>

__global__ void convertUint8ToFloat(const uint8_t *in, int size, float *out);

__global__ void convertFloatToUint8(const float *in, int size, uint8_t *out);

void launchConvertUint8ToFloat(const uint8_t *in, int size, float *out,
                               cudaStream_t stream);

void launchConvertFloatToUint8(const float *in, int size, uint8_t *out,
                               cudaStream_t stream);
