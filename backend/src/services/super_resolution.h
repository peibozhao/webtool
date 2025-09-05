#pragma once

#include "super_resolution_service.grpc.pb.h"
#include "super_resolution_service.pb.h"

class SuperResolution : public SuperResolutionService::Service {
public:
  grpc::Status Times4(grpc::ServerContext *context, const ImageRequest *request,
                      ImageResponse *response) override;

  grpc::Status Times2(grpc::ServerContext *context, const ImageRequest *request,
                      ImageResponse *response) override;

private:
};
