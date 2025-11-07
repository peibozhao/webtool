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

DEFINE_int32(grpc_port, 50051, "GRPC service port");
DEFINE_string(grpc_services, "", "Start GRPC service name");

class LoggingServerInterceptor : public grpc::experimental::Interceptor {
public:
  LoggingServerInterceptor(grpc::experimental::ServerRpcInfo *info) {
    rpc_info_ = info;
  }

  void Intercept(grpc::experimental::InterceptorBatchMethods *methods) {
    rpc_info_->type();
    if (methods->QueryInterceptionHookPoint(
            grpc::experimental::InterceptionHookPoints::PRE_SEND_STATUS)) {
      grpc::Status status = methods->GetSendStatus();
      if (status.ok()) {
        SPDLOG_INFO("GRPC method {} return ok", rpc_info_->method());
      } else {
        SPDLOG_ERROR("GRPC method {} return failed. code={}, message='{}'",
                     rpc_info_->method(), int(status.error_code()),
                     status.error_message());
      }
    }
    methods->Proceed();
  }

private:
  grpc::experimental::ServerRpcInfo *rpc_info_;
};

class LoggingServerInterceptorFactory
    : public grpc::experimental::ServerInterceptorFactoryInterface {
public:
  grpc::experimental::Interceptor *
  CreateServerInterceptor(grpc::experimental::ServerRpcInfo *info) {
    return new LoggingServerInterceptor(info);
  }
};

std::unique_ptr<grpc::Server> CreateGRPCServer() {
  SPDLOG_INFO("GRPC server is starting");
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder server_builder;
  server_builder.AddListeningPort(std::format("0.0.0.0:{}", FLAGS_grpc_port),
                                  grpc::InsecureServerCredentials());

  std::map<std::string, std::function<grpc::Service *()>> grpc_name_to_service =
      {
          {"copy", [] { return new CopyText(); }},
#ifdef USE_CUDA
          {"super_resolution", [] { return new SuperResolution(); }},
#endif
          {"qr_code", [] { return new QrCode(); }},
          {"map_pinning", [] { return new MapPinning(); }},
      };

  if (FLAGS_grpc_services == "all") {
    for (auto name_to_service : grpc_name_to_service) {
      SPDLOG_INFO("Register GRPC service {}", name_to_service.first);
      server_builder.RegisterService(name_to_service.second());
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
      server_builder.RegisterService(name_to_service_iter->second());
    }
  }
  SPDLOG_INFO("Build and start GRPC server");
  std::vector<
      std::unique_ptr<grpc::experimental::ServerInterceptorFactoryInterface>>
      interceptor_creators;
  interceptor_creators.emplace_back(new LoggingServerInterceptorFactory());
  server_builder.experimental().SetInterceptorCreators(
      std::move(interceptor_creators));
  return server_builder.BuildAndStart();
}
