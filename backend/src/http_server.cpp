#include "copy_service.grpc.pb.h"
#include "gflags/gflags.h"
#include "grpcpp/grpcpp.h"
#include "httplib.h"
#include "map_pinning.grpc.pb.h"
#include "qr_code.grpc.pb.h"
#include "spdlog/spdlog.h"
#include "src/proto/grpc/reflection/v1/reflection.grpc.pb.h"
#include "super_resolution_service.grpc.pb.h"
#include "utils/adapter.h"
#include "utils/log.h"
#include "utils/type_traits.h"
#include <ranges>

DECLARE_int32(grpc_port);

DEFINE_int32(http_port, 0, "Port for service");
DEFINE_string(grpc_servers, "", "gRPC servers address");

std::vector<std::string> GetGRPCServerAddresses() {
  static std::once_flag run_once;
  static std::vector<std::string> grpc_server_addrs;
  std::call_once(run_once, [] {
    auto servers_sview = FLAGS_grpc_servers | std::views::split(',');
    std::for_each(servers_sview.begin(), servers_sview.end(),
                  [](auto &&server) {
                    grpc_server_addrs.push_back(
                        std::string(server.begin(), server.end()));
                  });
  });
  return grpc_server_addrs;
}

std::shared_ptr<grpc::Channel>
GRPCServiceDiscovery(const std::vector<std::string> &addrs,
                     const std::string &cmd) {
  for (const std::string &addr : addrs) {
    std::unique_ptr<grpc::reflection::v1::ServerReflection::Stub>
        server_reflection = grpc::reflection::v1::ServerReflection::NewStub(
            grpc::CreateChannel(addr, grpc::InsecureChannelCredentials()));
    grpc::ClientContext context;
    auto server_info_rw_ptr = server_reflection->ServerReflectionInfo(&context);
    grpc::reflection::v1::ServerReflectionRequest request;
    request.set_list_services("*");
    if (!server_info_rw_ptr->Write(request) ||
        !server_info_rw_ptr->WritesDone()) {
      SPDLOG_WARN("GRPC server {} is stopped", addr);
      continue;
    }
    grpc::reflection::v1::ServerReflectionResponse response;
    while (server_info_rw_ptr->Read(&response)) {
      auto iter = std::find_if(
          response.list_services_response().service().begin(),
          response.list_services_response().service().end(),
          [&cmd](
              const grpc::reflection::v1::ServiceResponse &service_response) {
            return service_response.name() == cmd;
          });
      if (iter == response.list_services_response().service().end()) {
        continue;
      }
      SPDLOG_INFO("Discover service {} at address {}", cmd, addr);
      return grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
    }
  }
  throw std::runtime_error("服务器目前不可用");
}

template <typename ServiceType, auto Method>
void HTTPHandler(const httplib::Request &req, httplib::Response &res) {
  std::vector<std::string> grpc_server_addrs = GetGRPCServerAddresses();
  std::shared_ptr<grpc::Channel> channel =
      GRPCServiceDiscovery(grpc_server_addrs, ServiceType::service_full_name());

  typename GRPCTraits<decltype(Method)>::Request grpc_req;
  ConvertHTTPRequestToProto(req, &grpc_req);

  std::unique_ptr<typename ServiceType::Stub> stub =
      ServiceType::NewStub(channel);
  grpc::ClientContext context;
  typename GRPCTraits<decltype(Method)>::Response grpc_res;
  grpc::Status status = (stub.get()->*Method)(&context, grpc_req, &grpc_res);

  if (!status.ok()) {
    ConvertGRPCStatusToHTTPCode(status, res);
    return;
  }
  ConvertProtoToHTTPResponse(grpc_res, &res);
}

template <typename ServiceType, auto Method>
void RegisterPostHandler(std::unique_ptr<httplib::Server> &server_ptr,
                         const std::string &uri) {
  SPDLOG_INFO("Register HTTP POST {} handler", uri);
  server_ptr->Post(uri,
                   [&uri](const httplib::Request &req, httplib::Response &res) {
                     HTTPHandler<ServiceType, Method>(req, res);
                   });
}

template <typename ServiceType, auto Method>
void RegisterGetHandler(std::unique_ptr<httplib::Server> &server_ptr,
                        const std::string &uri) {
  SPDLOG_INFO("Register HTTP GET {} handler", uri);
  server_ptr->Get(uri,
                  [&uri](const httplib::Request &req, httplib::Response &res) {
                    HTTPHandler<ServiceType, Method>(req, res);
                  });
}

std::unique_ptr<httplib::Server> CreateHTTPServer() {
  SPDLOG_INFO("HTTP server starting");

  std::unique_ptr<httplib::Server> http_server_ptr =
      std::make_unique<httplib::Server>();
  http_server_ptr->set_logger(
      [](const httplib::Request &req, const httplib::Response &res) {
        SPDLOG_INFO("Receive HTTP request: {}", req);
        SPDLOG_INFO("Send HTTP response: {}", res);
      });
  http_server_ptr->set_error_handler(
      [](const httplib::Request &req, httplib::Response &res) {
        SPDLOG_WARN("Send HTTP error response: {}, body {}", res, res.body);
      });
  http_server_ptr->set_exception_handler([](const httplib::Request &req,
                                            httplib::Response &res,
                                            std::exception_ptr ep) {
    SPDLOG_WARN("Catch exception, rethrow exception");
    try {
      std::rethrow_exception(ep);
    } catch (const std::invalid_argument &e) {
      SPDLOG_WARN("Catch invalid_argument exception {}", e.what());
      res.status = httplib::StatusCode::BadRequest_400;
      res.body = e.what();
    } catch (const std::exception &e) {
      SPDLOG_WARN("Catch exception {}", e.what());
      res.status = httplib::StatusCode::InternalServerError_500;
      res.body = e.what();
    }
  });

#define REGISTER_HTTP_HANDLER(Method, Uri, Service, RPC)                       \
  Register##Method##Handler<Service, &Service::Stub::RPC>(http_server_ptr, Uri);

  SPDLOG_INFO("Set copy service handler");
  REGISTER_HTTP_HANDLER(Post, "/api/copy/submit", CopyService, Submit);
  REGISTER_HTTP_HANDLER(Get, "/api/copy/retrieve", CopyService, Retrieve);

  SPDLOG_INFO("Set super resolution service handler");
  REGISTER_HTTP_HANDLER(Post, "/api/super_resolution", SuperResolutionService,
                        Times4);

  SPDLOG_INFO("Set QR code service handler");
  REGISTER_HTTP_HANDLER(Post, "/api/qr_code/parse", QrCodeService, ParseText);
  REGISTER_HTTP_HANDLER(Post, "/api/qr_code/generate", QrCodeService,
                        GenerateImage);

  SPDLOG_INFO("Set map pinning service handler");
  REGISTER_HTTP_HANDLER(Post, "/api/map_pinning/upload", MapPinningService,
                        Upload);
  REGISTER_HTTP_HANDLER(Get, "/api/map_pinning/download", MapPinningService,
                        Download);

  SPDLOG_INFO("Set check healthy service handler");
  http_server_ptr->Get("/api/check_healthy",
                       [](const httplib::Request &req, httplib::Response &res) {
                         res.set_content("Hello", "text/plain; charset=utf-8");
                         SPDLOG_INFO("Receive check healthy request");
                       });

  SPDLOG_INFO("Return HTTP server pointer");
  return http_server_ptr;
}
