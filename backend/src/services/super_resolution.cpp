#ifdef USE_CUDA

#include "super_resolution.h"
#include "spdlog/spdlog.h"
#include "utils/cuda_utils.h"
#include "utils/log.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nvjpeg.h>
#include <opencv2/opencv.hpp>

#define NVJPEG_CHECK(expression)                                               \
  {                                                                            \
    nvjpegStatus_t status = expression;                                        \
    if (status != NVJPEG_STATUS_SUCCESS) {                                     \
      SPDLOG_ERROR("nvjpeg function return failed, status is {}",              \
                   int(status));                                               \
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "TODO");         \
    }                                                                          \
  }

#define CUDA_CHECK(expression)                                                 \
  {                                                                            \
    cudaError status = expression;                                             \
    if (status != cudaSuccess) {                                               \
      SPDLOG_ERROR("cuda function return failed, status is {}", int(status));  \
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "TODO");         \
    }                                                                          \
  }

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // suppress info-level messages
    if (severity >= Severity::kINFO)
      SPDLOG_INFO("{}", msg);
  }
} gLogger;

SuperResolution::SuperResolution() {
  CUDA_ASSERT(cudaStreamCreate(&stream_));

  // Load engine file
  std::filesystem::path fpath("../models/realesrgan-x4.engine");
  std::ifstream ifs(fpath, std::ios::binary);
  int fsize = std::filesystem::file_size(fpath);
  SPDLOG_INFO("Tensorrt engine file {} size {}", fpath.string(), fsize);
  std::string fcontent(fsize, '.');
  ifs.read(fcontent.data(), fcontent.size());
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  ASSERT(runtime != nullptr);
  engine_ = runtime->deserializeCudaEngine(fcontent.data(), fcontent.size());
  ASSERT(engine_ != nullptr);
  infer_context_ = engine_->createExecutionContext();
  ASSERT(infer_context_ != nullptr);

  NVJPEG_ASSERT(nvjpegCreateSimple(&nvjpeg_handle_));
  delete runtime;
}

grpc::Status SuperResolution::Times4(grpc::ServerContext *context,
                                     const ImageRequest *request,
                                     ImageResponse *response) {
  auto start_time = std::chrono::system_clock::now();

  nvjpegJpegState_t jpeg_state;
  NVJPEG_ASSERT(nvjpegJpegStateCreate(nvjpeg_handle_, &jpeg_state));

  int components;
  nvjpegChromaSubsampling_t subsample;
  int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
  NVJPEG_CHECK(nvjpegGetImageInfo(
      nvjpeg_handle_, (const unsigned char *)request->image().data(),
      request->image().size(), &components, &subsample, widths, heights));

  int width = widths[0], height = heights[0];
  SPDLOG_INFO("Input image width {} height {}", width, height);

  uint8_t *nvjpeg_image_buffer;
  CUDA_ASSERT(
      cudaMallocAsync(&nvjpeg_image_buffer, width * height * 3, stream_));
  nvjpegImage_t nvjpeg_image;
  nvjpeg_image.pitch[0] = width;
  nvjpeg_image.channel[0] = nvjpeg_image_buffer;
  nvjpeg_image.pitch[1] = width;
  nvjpeg_image.channel[1] = nvjpeg_image_buffer + width * height;
  nvjpeg_image.pitch[2] = width;
  nvjpeg_image.channel[2] = nvjpeg_image_buffer + width * height * 2;

  NVJPEG_CHECK(nvjpegDecode(
      nvjpeg_handle_, jpeg_state,
      (const unsigned char *)request->image().data(), request->image().size(),
      nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB, &nvjpeg_image, stream_));

  nvinfer1::Dims4 input_dims{1, 3, height, width};
  int input_size =
      std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 4,
                      std::multiplies<int64_t>{});
  SPDLOG_INFO("Engine input memory {}", input_size);
  ASSERT(infer_context_->setInputShape("input", input_dims));
  ASSERT(infer_context_->allInputDimensionsSpecified());
  float *input_dev_ptr;
  CUDA_ASSERT(cudaMallocAsync(&input_dev_ptr, input_size, stream_));
  ASSERT(infer_context_->setInputTensorAddress("input", input_dev_ptr));

  launchConvertUint8ToFloat(nvjpeg_image_buffer, width * height * 3,
                            input_dev_ptr, stream_);

  int output_width = width * 4, output_height = height * 4;
  float *output_dev_ptr;
  nvinfer1::Dims4 output_dims{1, 3, output_height, output_width};
  int output_size =
      std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 4,
                      std::multiplies<int64_t>{});
  CUDA_ASSERT(cudaMallocAsync(&output_dev_ptr, output_size, stream_));
  ASSERT(infer_context_->setOutputTensorAddress("output", output_dev_ptr));

  ASSERT(infer_context_->enqueueV3(stream_));

  uint8_t *output_image_dev_ptr;
  CUDA_ASSERT(cudaMallocAsync(&output_image_dev_ptr,
                              output_width * output_height * 3, stream_));

  launchConvertFloatToUint8(output_dev_ptr, output_width * output_height * 3,
                            output_image_dev_ptr, stream_);

  nvjpegEncoderState_t nvjpeg_encoder_state;
  NVJPEG_ASSERT(
      nvjpegEncoderStateCreate(nvjpeg_handle_, &nvjpeg_encoder_state, stream_));

  nvjpegEncoderParams_t encoder_params;
  NVJPEG_ASSERT(
      nvjpegEncoderParamsCreate(nvjpeg_handle_, &encoder_params, stream_));
  NVJPEG_ASSERT(nvjpegEncoderParamsSetQuality(encoder_params, 80, stream_));
  NVJPEG_ASSERT(nvjpegEncoderParamsSetSamplingFactors(encoder_params,
                                                      NVJPEG_CSS_420, stream_));

  nvjpegImage_t encode_image;
  memset(&encode_image, 0, sizeof(encode_image));
  encode_image.pitch[0] = output_width;
  encode_image.channel[0] = output_image_dev_ptr;
  encode_image.pitch[1] = output_width;
  encode_image.channel[1] = output_image_dev_ptr + output_height * output_width;
  encode_image.pitch[2] = output_width;
  encode_image.channel[2] =
      output_image_dev_ptr + output_height * output_width * 2;

  NVJPEG_ASSERT(nvjpegEncodeImage(
      nvjpeg_handle_, nvjpeg_encoder_state, encoder_params, &encode_image,
      NVJPEG_INPUT_RGB, output_width, output_height, stream_));

  size_t encode_jpeg_size;
  NVJPEG_ASSERT(nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle_, nvjpeg_encoder_state, NULL, &encode_jpeg_size, stream_));
  SPDLOG_INFO("Encode image size {}", encode_jpeg_size);
  response->mutable_image()->resize(encode_jpeg_size);
  NVJPEG_ASSERT(nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle_, nvjpeg_encoder_state,
      (unsigned char *)response->mutable_image()->data(), &encode_jpeg_size,
      stream_));

  cudaStreamSynchronize(stream_);
  CUDA_CHECK(cudaGetLastError());

  cudaFreeAsync(input_dev_ptr, stream_);
  cudaFreeAsync(nvjpeg_image_buffer, stream_);
  cudaFreeAsync(output_dev_ptr, stream_);
  cudaFreeAsync(output_image_dev_ptr, stream_);
  nvjpegJpegStateDestroy(jpeg_state);
  nvjpegEncoderStateDestroy(nvjpeg_encoder_state);
  nvjpegEncoderParamsDestroy(encoder_params);

  auto end_time = std::chrono::system_clock::now();
  SPDLOG_INFO("Super resolution cost {}",
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  end_time - start_time));
  return grpc::Status::OK;
}

grpc::Status SuperResolution::Times2(grpc::ServerContext *context,
                                     const ImageRequest *request,
                                     ImageResponse *response) {
  return grpc::Status::OK;
}

#endif
