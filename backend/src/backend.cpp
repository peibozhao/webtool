#include "gflags/gflags.h"
#include "grpcpp/grpcpp.h"
#include "httplib.h"
#include "services/qr_code.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "utils/utils.h"

std::unique_ptr<grpc::Server> CreateGRPCServer();
std::unique_ptr<httplib::Server> CreateHTTPServer();

DECLARE_string(grpc_services);
DECLARE_int32(http_port);

DEFINE_bool(debug, false, "Enable debug mode");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  spdlog::flush_every(std::chrono::seconds(1));

  std::filesystem::path log_fpath =
      GetLogDirectory() / (GetProcessName() + ".log");
  std::shared_ptr<spdlog::sinks::sink> file_sink =
      std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
          log_fpath, 1024 * 1024 * 5, 10);
  std::vector<std::shared_ptr<spdlog::sinks::sink>> sinks{file_sink};
  if (FLAGS_debug) {
    std::shared_ptr<spdlog::sinks::sink> console_sink =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
  }
  std::shared_ptr<spdlog::logger> logger = std::make_shared<spdlog::logger>(
      GetProcessName(), sinks.begin(), sinks.end());
  spdlog::set_default_logger(logger);

  std::unique_ptr<grpc::Server> grpc_server_ptr;
  if (!FLAGS_grpc_services.empty()) {
    grpc_server_ptr = CreateGRPCServer();
  }
  std::unique_ptr<httplib::Server> http_server_ptr;
  if (FLAGS_http_port > 0) {
    http_server_ptr = CreateHTTPServer();
  }

  if (grpc_server_ptr != nullptr) {
    std::thread([&grpc_server_ptr] {
      SPDLOG_INFO("Start GRPC server");
      grpc_server_ptr->Wait();
      SPDLOG_ERROR("GRPC server stoped");
    }).detach();
  }
  if (http_server_ptr != nullptr) {
    std::thread([&http_server_ptr] {
      SPDLOG_INFO("Start http server, listen port {}", FLAGS_http_port);
      http_server_ptr->listen("0.0.0.0", FLAGS_http_port);
      SPDLOG_ERROR("HTTP server stoped");
    }).detach();
  }

  if (grpc_server_ptr == nullptr && http_server_ptr == nullptr) {
    SPDLOG_ERROR("HTTP server and GRPC server is all disabled, exit");
    return 1;
  }
  SPDLOG_INFO("Block for server");
  std::this_thread::sleep_for(std::chrono::years(1));
  return 0;
}
