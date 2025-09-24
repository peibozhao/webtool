
#include "gflags/gflags.h"
#include "grpcpp/ext/proto_server_reflection_plugin.h"
#include "grpcpp/grpcpp.h"
#include "services/copy_text.h"
#include "services/qr_code.h"
#include "spdlog/spdlog.h"
#ifdef USE_CUDA
#include "services/super_resolution.h"
#endif

DEFINE_int32(grpc_port, 50051, "gRPC service port");
DEFINE_string(grpc_services, "", "Start gRPC service name");

std::unique_ptr<grpc::Server> CreateGrpcServer() {
  SPDLOG_INFO("gRPC server is starting");
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder server_builder;
  server_builder.AddListeningPort(std::format("0.0.0.0:{}", FLAGS_grpc_port),
                                  grpc::InsecureServerCredentials());

  for (const auto &service_range :
       FLAGS_grpc_services | std::views::split(',')) {
    std::string_view service_name = std::string_view(
        &*service_range.begin(), std::ranges::distance(service_range));

    if (service_name == "copy") {
      CopyText *copy = new CopyText();
      server_builder.RegisterService(copy);
    } else if (service_name == "super_resolution") {
#ifdef USE_CUDA
      SuperResolution *super_resolution = new SuperResolution();
      server_builder.RegisterService(super_resolution);
#else
      SPDLOG_ERROR("Service {} need cuda, but current runtime environment "
                   "don't support",
                   service_name);
      abort();
#endif
    } else if (service_name == "qr_code") {
      QrCode *qr_code = new QrCode();
      server_builder.RegisterService(qr_code);
    } else {
      SPDLOG_WARN("Unknown service name {}", service_name);
      continue;
    }
    SPDLOG_INFO("Register {} grpc service", service_name);
  }

  SPDLOG_INFO("Build and start grpc server");
  return server_builder.BuildAndStart();
}
