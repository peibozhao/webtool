#pragma once

#include "super_resolution_service.grpc.pb.h"
#include "super_resolution_service.pb.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <nvjpeg.h>

class SuperResolution : public SuperResolutionService::Service {
public:
  SuperResolution();

  grpc::Status Times4(grpc::ServerContext *context, const ImageRequest *request,
                      ImageResponse *response) override;

  grpc::Status Times2(grpc::ServerContext *context, const ImageRequest *request,
                      ImageResponse *response) override;

private:
  cudaStream_t stream_;
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *infer_context_;
  nvjpegHandle_t nvjpeg_handle_;
};

