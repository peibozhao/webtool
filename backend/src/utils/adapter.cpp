#include "adapter.h"

void ConvertGRPCStatusToHTTPCode(const grpc::Status &grpc_status,
                                 httplib::Response &res) {
  if (grpc_status.ok()) {
    return;
  }
  SPDLOG_ERROR("GRPC call return failed. code={}, message='{}'",
               int(grpc_status.error_code()), grpc_status.error_message());
  int grpc_errcode = grpc_status.error_code();
  switch (grpc_status.error_code()) {
  case grpc::StatusCode::UNAUTHENTICATED:
    res.status = 403;
    res.body = "密码错误";
    break;
  case grpc::StatusCode::NOT_FOUND:
    res.status = 404;
    res.body = "资源不存在";
    break;
  default:
    res.status = 500;
    res.body = "未知错误";
    break;
  }
}
