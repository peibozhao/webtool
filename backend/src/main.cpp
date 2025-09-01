
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_builder.h"
#include "httplib.h"
#include "services/copy_text.h"
#include "spdlog/spdlog.h"
#include <format>

grpc::ByteBuffer MessageToByteBuffer(const google::protobuf::Message &msg) {
  std::string serialized;
  if (!msg.SerializeToString(&serialized)) {
    throw std::runtime_error("Failed to serialize message");
  }

  grpc::Slice slice(serialized.data(), serialized.size());
  std::vector<grpc::Slice> slices;
  slices.push_back(std::move(slice));

  grpc::ByteBuffer buffer(&slices[0], slices.size());
  return buffer;
}

std::unordered_map<std::string, std::string> path_params;

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
                          "headers({}), body({})",
                          request.remote_addr, request.remote_port,
                          request.method, request.path, request.params,
                          request.headers, request.body);
  }
};

template <> struct std::formatter<httplib::Response> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Response &response, format_context &fc) const {
    return std::format_to(fc.out(), "staus({}), header({}), body({})",
                          response.status, response.headers, response.body);
  }
};

int main(int argc, char *argv[]) {
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

  // gRPC server
  std::thread grcp_thread([] {
    CopyText copy;
    grpc::ServerBuilder server_builder;
    server_builder.AddListeningPort("0.0.0.0:50051",
                                    grpc::InsecureServerCredentials());
    server_builder.RegisterService(&copy);
    std::unique_ptr<grpc::Server> server = server_builder.BuildAndStart();
    server->Wait();
  });

  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "ipv4:127.0.0.1:50051", grpc::InsecureChannelCredentials());

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

  SPDLOG_INFO("Set check healthy service handler");
  http_server.Get("/api/check_healthy",
                  [](const httplib::Request &req, httplib::Response &res) {
                    res.set_content("Hello", "text/plain; charset=utf-8");
                    SPDLOG_INFO("Receive check healthy request");
                  });

  // Process
  SPDLOG_INFO("Server is starting");
  http_server.listen("0.0.0.0", 8080);
  SPDLOG_WARN("Server is stoped");
  return 0;
}
