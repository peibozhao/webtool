
#include "super_resolution.h"

grpc::Status SuperResolution::Times4(grpc::ServerContext *context,
                                     const ImageRequest *request,
                                     ImageResponse *response) {
  return grpc::Status::OK;
}

grpc::Status SuperResolution::Times2(grpc::ServerContext *context,
                                     const ImageRequest *request,
                                     ImageResponse *response) {
  return grpc::Status::OK;
}
