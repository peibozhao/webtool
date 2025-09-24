
#include "copy_service.grpc.pb.h"
#include "gflags/gflags.h"
#include "grpcpp/grpcpp.h"
#include "httplib.h"
#include "utils/log.h"
#include "qr_code.grpc.pb.h"
#include "spdlog/spdlog.h"
#include "src/proto/grpc/reflection/v1/reflection.grpc.pb.h"
#include "super_resolution_service.grpc.pb.h"
#include <ranges>

DEFINE_int32(http_port, 0, "Port for service");
DEFINE_string(grpc_servers, "127.0.0.1:50051", "gRPC servers address");

std::shared_ptr<grpc::Channel>
GrpcServiceDiscovery(const std::vector<std::string> &addrs,
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
      SPDLOG_WARN("Grpc server {} is stopped", addr);
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
  throw std::runtime_error("Grpc server is invalid");
}

std::unique_ptr<httplib::Server> CreateHttpServer() {
  SPDLOG_INFO("HTTP server starting");
  auto servers_sview = FLAGS_grpc_servers | std::views::split(',');
  std::vector<std::string> addrs;
  std::for_each(servers_sview.begin(), servers_sview.end(),
                [&addrs](auto &&server) {
                  addrs.push_back(std::string(server.begin(), server.end()));
                });

  std::unique_ptr<httplib::Server> http_server_ptr =
      std::make_unique<httplib::Server>();
  http_server_ptr->set_logger(
      [](const httplib::Request &req, const httplib::Response &res) {
        SPDLOG_INFO("Receive http request: {}, response {}", req, res);
      });
  http_server_ptr->set_error_handler(
      [](const httplib::Request &req, httplib::Response &res) {
        SPDLOG_WARN("Receive invalid http request: {}. response {}", req, res);
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
      res.set_header("EXCEPTION_WHAT", e.what());
    } catch (const std::exception &e) {
      SPDLOG_WARN("Catch exception {}", e.what());
      res.status = httplib::StatusCode::InternalServerError_500;
      res.set_header("EXCEPTION_WHAT", e.what());
    }
  });

  httplib::Headers default_headers;
  default_headers.insert({"Access-Control-Allow-Origin", "*"});
  http_server_ptr->set_default_headers(default_headers);

  SPDLOG_INFO("Set copy service handler");
  http_server_ptr->Post("/api/copy/submit", [addrs](const httplib::Request &req,
                                                    httplib::Response &res) {
    std::shared_ptr<grpc::Channel> channel =
        GrpcServiceDiscovery(addrs, CopyService::service_full_name());
    std::unique_ptr<CopyService::Stub> stub = CopyService::NewStub(channel);
    SubmitRequest req_pb;
    req_pb.set_text(req.get_param_value("text"));
    grpc::ClientContext context;
    SubmitResponse res_pb;
    grpc::Status status = stub->Submit(&context, req_pb, &res_pb);
    if (!status.ok()) {
      SPDLOG_WARN("gRPC call Submmit failed. code={}, message={}",
                  int(status.error_code()), status.error_message());
      return;
    }
    res.set_content(std::format(R"({{"code": "{}"}})", res_pb.code()),
                    "application/json");
  });

  http_server_ptr->Get(
      "/api/copy/retrieve",
      [addrs](const httplib::Request &req, httplib::Response &res) {
        std::shared_ptr<grpc::Channel> channel =
            GrpcServiceDiscovery(addrs, CopyService::service_full_name());
        std::unique_ptr<CopyService::Stub> stub = CopyService::NewStub(channel);
        RetrieveRequest req_pb;
        req_pb.set_code(req.get_param_value("code"));
        grpc::ClientContext context;
        RetrieveResponse res_pb;
        grpc::Status status = stub->Retrieve(&context, req_pb, &res_pb);
        if (!status.ok()) {
          SPDLOG_WARN("gRPC call Retrieve failed. code={}, message={}",
                      int(status.error_code()), status.error_message());
          return;
        }
        res.set_content(std::format(R"({{"text": "{}"}})", res_pb.text()),
                        "application/json");
      });

  SPDLOG_INFO("Set super resolution service handler");
  http_server_ptr->Post(
      "/api/super_resolution",
      [addrs](const httplib::Request &req, httplib::Response &res) {
        const httplib::FormData file_form_data = req.form.get_file("image");
        if (file_form_data.content.empty()) {
          throw std::invalid_argument("File content is empty");
        }

        std::shared_ptr<grpc::Channel> channel = GrpcServiceDiscovery(
            addrs, SuperResolutionService::service_full_name());
        std::unique_ptr<SuperResolutionService::Stub> stub =
            SuperResolutionService::NewStub(channel);
        ImageRequest image_request;
        image_request.set_image(file_form_data.content);
        grpc::ClientContext context;
        ImageResponse res_pb;
        grpc::Status status = stub->Times4(&context, image_request, &res_pb);
        res.set_content(res_pb.image(), "image/jpg");
        SPDLOG_INFO("Response image size {}", res_pb.image().size());
      });

  SPDLOG_INFO("Set QR code service handler");
  http_server_ptr->Post("/api/qr_code/parse", [addrs](
                                                  const httplib::Request &req,
                                                  httplib::Response &res) {
    const httplib::FormData file_form_data = req.form.get_file("image");
    if (file_form_data.content.empty()) {
      throw std::invalid_argument("File content is empty");
    }

    std::shared_ptr<grpc::Channel> channel =
        GrpcServiceDiscovery(addrs, QrCodeService::service_full_name());
    std::unique_ptr<QrCodeService::Stub> stub = QrCodeService::NewStub(channel);
    grpc::ClientContext context;
    ParseTextRequest parse_request;
    ParseTextResponse parse_response;
    parse_request.set_image(file_form_data.content);
    grpc::Status grpc_status =
        stub->ParseText(&context, parse_request, &parse_response);
    res.set_content(std::format(R"({{"text":"{}"}})", parse_response.text()),
                    "application/json");
  });
  http_server_ptr->Post(
      "/api/qr_code/generate",
      [addrs](const httplib::Request &req, httplib::Response &res) {
        std::shared_ptr<grpc::Channel> channel =
            GrpcServiceDiscovery(addrs, QrCodeService::service_full_name());
        std::unique_ptr<QrCodeService::Stub> stub =
            QrCodeService::NewStub(channel);
        GenerateImageRequest req_pb;
        req_pb.set_text(req.get_param_value("text"));
        grpc::ClientContext context;
        GenerateImageResponse res_pb;
        grpc::Status grpc_status =
            stub->GenerateImage(&context, req_pb, &res_pb);
        res.set_content(res_pb.image(), "image/jpg");
      });

  SPDLOG_INFO("Set check healthy service handler");
  http_server_ptr->Get("/api/check_healthy",
                       [](const httplib::Request &req, httplib::Response &res) {
                         res.set_content("Hello", "text/plain; charset=utf-8");
                         SPDLOG_INFO("Receive check healthy request");
                       });

  SPDLOG_INFO("Return HTTP server pointer");
  return http_server_ptr;
}
