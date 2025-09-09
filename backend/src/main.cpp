
#include "gflags/gflags.h"
#include "google/protobuf/descriptor.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_builder.h"
#include "httplib.h"
#include "services/copy_text.h"
#include "services/super_resolution.h"
#include "services/qr_code.h"
#include "spdlog/spdlog.h"
#include <format>

DEFINE_int32(port, 8080, "Port for service");
DEFINE_string(services, "", "gRPC service name");

static constexpr int kGrpcPort = 50051;

template <>
struct std::formatter<std::unordered_map<std::string, std::string>> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const std::unordered_map<std::string, std::string> &path_params,
              format_context &fc) const {
    if (path_params.empty()) {
      return fc.out();
    }
    auto iter = path_params.begin();
    for (int i = 0; i < path_params.size() - 1; ++i) {
      std::format_to(fc.out(), "{}: {}, ", iter->first, iter->second);
      ++iter;
    }
    return std::format_to(fc.out(), "{}: {}", iter->first, iter->second);
  }
};

template <> struct std::formatter<httplib::Params> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Params &params, format_context &fc) const {
    if (params.empty()) {
      return fc.out();
    }
    std::for_each_n(params.begin(), params.size() - 1,
                    [&fc](const std::pair<std::string, std::string> &param) {
                      std::format_to(fc.out(), "{}: {}, ", param.first,
                                     param.second);
                    });
    return std::format_to(fc.out(), "{}: {}", params.rbegin()->first,
                          params.rbegin()->second);
  }
};

template <> struct std::formatter<httplib::Headers> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Headers &headers, format_context &fc) const {
    if (headers.empty()) {
      return fc.out();
    }
    auto iter = headers.begin();
    for (int i = 0; i < headers.size() - 1; ++i) {
      std::format_to(fc.out(), "{}: {}, ", iter->first, iter->second);
      ++iter;
    }
    return std::format_to(fc.out(), "{}: {}", iter->first, iter->second);
  }
};

template <> struct std::formatter<httplib::Request> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Request &request, format_context &fc) const {
    return std::format_to(fc.out(),
                          "address({}:{}), method({}), path({}), params({}), "
                          "headers({})",
                          request.remote_addr, request.remote_port,
                          request.method, request.path, request.params,
                          request.headers);
  }
};

template <> struct std::formatter<httplib::Response> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Response &response, format_context &fc) const {
    return std::format_to(fc.out(), "staus({}), header({})", response.status,
                          response.headers);
  }
};

void StartGrpcServer() {
  SPDLOG_INFO("gRPC server is starting");
  grpc::ServerBuilder server_builder;
  server_builder.AddListeningPort(std::format("0.0.0.0:{}", kGrpcPort),
                                  grpc::InsecureServerCredentials());

  for (const auto &service_range : FLAGS_services | std::views::split(',')) {
    std::string_view service_name = std::string_view(
        &*service_range.begin(), std::ranges::distance(service_range));
    const google::protobuf::DescriptorPool *desc_pool =
        google::protobuf::DescriptorPool::generated_pool();
    desc_pool->FindServiceByName(service_name);

    if (service_name == "copy") {
      CopyText *copy = new CopyText();
      server_builder.RegisterService(copy);
    } else if (service_name == "super_resolution") {
      SuperResolution *super_resolution = new SuperResolution();
      server_builder.RegisterService(super_resolution);
    } else if (service_name == "qr_code") {
      QrCode *qr_code = new QrCode();
      server_builder.RegisterService(qr_code);
    } else {
      SPDLOG_WARN("Unknown service name {}", service_name);
      continue;
    }
    SPDLOG_INFO("Register {} grpc service", service_name);
  }

  std::unique_ptr<grpc::Server> server = server_builder.BuildAndStart();
  server->Wait();
  SPDLOG_ERROR("gRPC server is stopped");
}

void StartHttpServer() {
  SPDLOG_INFO("HTTP server starting");
  httplib::Server http_server;
  http_server.set_logger(
      [](const httplib::Request &req, const httplib::Response &res) {
        SPDLOG_INFO("Receive http request: {}, response {}", req, res);
      });
  http_server.set_error_handler(
      [](const httplib::Request &req, httplib::Response &res) {
        SPDLOG_WARN("Receive invalid http request: {}. response {}", req, res);
      });
  http_server.set_exception_handler([](const httplib::Request &req,
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
  http_server.set_default_headers(default_headers);

  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(std::format("ipv4:127.0.0.1:{}", kGrpcPort),
                          grpc::InsecureChannelCredentials());

  SPDLOG_INFO("Set copy service handler");
  http_server.Post("/api/copy/submit", [&channel](const httplib::Request &req,
                                                  httplib::Response &res) {
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

  http_server.Get("/api/copy/retrieve", [channel](const httplib::Request &req,
                                                  httplib::Response &res) {
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
  http_server.Post(
      "/api/super_resolution",
      [channel](const httplib::Request &req, httplib::Response &res) {
        if (req.form.files.empty()) {
          throw std::invalid_argument("No files uploaded");
        }

        const httplib::FormData file_form_data = req.form.get_file("image");
        if (file_form_data.content.empty()) {
          throw std::invalid_argument("File content is empty");
        }
        // TODO
        res.set_content(file_form_data.content, "image/jpg");
      });

  SPDLOG_INFO("Set QR code service handler");
  http_server.Post("/api/qr_code/parse", [channel](const httplib::Request &req,
                                                   httplib::Response &res) {
    if (req.form.files.empty()) {
      throw std::invalid_argument("No files uploaded");
    }

    const httplib::FormData file_form_data = req.form.get_file("image");
    if (file_form_data.content.empty()) {
      throw std::invalid_argument("File content is empty");
    }
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
  http_server.Post("/api/qr_code/generate", [channel](
                                                const httplib::Request &req,
                                                httplib::Response &res) {
    std::unique_ptr<QrCodeService::Stub> stub = QrCodeService::NewStub(channel);
    GenerateImageRequest req_pb;
    req_pb.set_text(req.get_param_value("text"));
    grpc::ClientContext context;
    GenerateImageResponse res_pb;
    grpc::Status grpc_status = stub->GenerateImage(&context, req_pb, &res_pb);
    res.set_content(res_pb.image(), "image/jpg");
  });

  SPDLOG_INFO("Set check healthy service handler");
  http_server.Get("/api/check_healthy",
                  [](const httplib::Request &req, httplib::Response &res) {
                    res.set_content("Hello", "text/plain; charset=utf-8");
                    SPDLOG_INFO("Receive check healthy request");
                  });

  http_server.listen("0.0.0.0", FLAGS_port);
  SPDLOG_ERROR("HTTP server is stoped");
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_services.empty()) {
    std::thread([] {
      SPDLOG_INFO("gRPC server thread is running");
      StartGrpcServer();
    }).detach();
  }
  if (FLAGS_port > 0) {
    std::thread([] {
      SPDLOG_INFO("HTTP server is runnign, listen port {}", FLAGS_port);
      StartHttpServer();
    }).detach();
  }

  SPDLOG_INFO("Block for server");
  std::this_thread::sleep_for(std::chrono::years(1));
  return 0;
}
