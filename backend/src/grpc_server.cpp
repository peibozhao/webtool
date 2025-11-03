#include "gflags/gflags.h"
#include "grpcpp/ext/proto_server_reflection_plugin.h"
#include "grpcpp/grpcpp.h"
#include "services/copy_text.h"
#include "services/map_pinning.h"
#include "services/qr_code.h"
#include "spdlog/spdlog.h"
#ifdef USE_CUDA
#include "services/super_resolution.h"
#endif

DEFINE_int32(grpc_port, 50051, "gRPC service port");
DEFINE_string(grpc_services, "", "Start gRPC service name");

namespace {
std::map<std::string, grpc::Service *> grpc_name_to_service = {
    {"copy", new CopyText()},
#ifdef USE_CUDA
    {"super_resolution", new SuperResolutionService()},
#endif
    {"qr_code", new QrCode()},
    {"map_pinning", new MapPinning()},
};
} // namespace

std::unique_ptr<grpc::Server> CreateGRPCServer() {
  SPDLOG_INFO("gRPC server is starting");
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder server_builder;
  server_builder.AddListeningPort(std::format("0.0.0.0:{}", FLAGS_grpc_port),
                                  grpc::InsecureServerCredentials());

  if (FLAGS_grpc_services == "all") {
    for (auto name_to_service : grpc_name_to_service) {
      SPDLOG_INFO("Register GRPC service {}", name_to_service.first);
      server_builder.RegisterService(name_to_service.second);
    }
  } else {
    for (const auto &service_range :
         FLAGS_grpc_services | std::views::split(',')) {
      std::string_view service_name = std::string_view(
          &*service_range.begin(), std::ranges::distance(service_range));

      auto name_to_service_iter =
          grpc_name_to_service.find(std::string(service_name));
      if (name_to_service_iter == grpc_name_to_service.end()) {
        SPDLOG_ERROR("Unknown service name {}", service_name);
        return nullptr;
      }
      SPDLOG_INFO("Register GRPC service {}", service_name);
      server_builder.RegisterService(name_to_service_iter->second);
    }
  }
  SPDLOG_INFO("Build and start GRPC server");
  return server_builder.BuildAndStart();
}
