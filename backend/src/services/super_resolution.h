#pragma once

#include "super_resolution_service.grpc.pb.h"
#include "super_resolution_service.pb.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <nvjpeg.h>

class SuperResolution : public SuperResolutionService::Service {
public:
  SuperResolution();

  grpc::Status Times4(grpc::ServerContext *context, const Times4Request *request,
                      Times4Response *response) override;

private:
  cudaStream_t stream_;
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *infer_context_;
  nvjpegHandle_t nvjpeg_handle_;
  nvinfer1::Dims min_dims_;
  nvinfer1::Dims max_dims_;
};

