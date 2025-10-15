
#include "gflags/gflags.h"
#include "grpcpp/grpcpp.h"
#include "httplib.h"
#include "services/qr_code.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"

std::unique_ptr<grpc::Server> CreateGrpcServer();
std::unique_ptr<httplib::Server> CreateHttpServer();

DECLARE_string(grpc_services);
DECLARE_int32(http_port);

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  auto logger =
      spdlog::rotating_logger_mt("backend", "backend.log", 1024 * 1024 * 5, 10);
  spdlog::set_default_logger(logger);

  std::unique_ptr<grpc::Server> grpc_server_ptr;
  if (!FLAGS_grpc_services.empty()) {
    grpc_server_ptr = CreateGrpcServer();
  }
  std::unique_ptr<httplib::Server> http_server_ptr;
  if (FLAGS_http_port > 0) {
    http_server_ptr = CreateHttpServer();
  }

  if (grpc_server_ptr != nullptr) {
    std::thread([&grpc_server_ptr] {
      SPDLOG_INFO("Start grpc server");
      grpc_server_ptr->Wait();
      SPDLOG_ERROR("Grpc server stoped");
    }).detach();
  }
  if (http_server_ptr != nullptr) {
    std::thread([&http_server_ptr] {
      SPDLOG_INFO("Start http server, listen port {}", FLAGS_http_port);
      http_server_ptr->listen("0.0.0.0", FLAGS_http_port);
      SPDLOG_ERROR("Http server stoped");
    }).detach();
  }

  if (grpc_server_ptr == nullptr && http_server_ptr == nullptr) {
    SPDLOG_ERROR("Http server and grpc server is all disabled, exit");
    return 1;
  }
  SPDLOG_INFO("Block for server");
  std::this_thread::sleep_for(std::chrono::years(1));
  return 0;
}
