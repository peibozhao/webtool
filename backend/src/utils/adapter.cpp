#include "adapter.h"

namespace {
static const std::unordered_map<grpc::StatusCode, int> kGRPC2HTTP = {
    {grpc::StatusCode::OK, 200},
    {grpc::StatusCode::UNKNOWN, 500},
    {grpc::StatusCode::INVALID_ARGUMENT, 400},
    {grpc::StatusCode::NOT_FOUND, 404},
    {grpc::StatusCode::ALREADY_EXISTS, 409},
    {grpc::StatusCode::PERMISSION_DENIED, 403},
    {grpc::StatusCode::UNAUTHENTICATED, 401},
    {grpc::StatusCode::RESOURCE_EXHAUSTED, 429},
    {grpc::StatusCode::OUT_OF_RANGE, 400},
    {grpc::StatusCode::UNIMPLEMENTED, 501},
    {grpc::StatusCode::INTERNAL, 500},
    {grpc::StatusCode::UNAVAILABLE, 503},
};
}

void ConvertGRPCStatusToHTTPCode(const grpc::Status &grpc_status,
                                 httplib::Response &res) {
  if (grpc_status.ok()) {
    return;
  }
  SPDLOG_ERROR("GRPC call return failed. code={}, message='{}'",
               int(grpc_status.error_code()), grpc_status.error_message());
  auto iter = kGRPC2HTTP.find(grpc_status.error_code());
  res.status = iter == kGRPC2HTTP.end() ? 500 : iter->second;
  res.body = grpc_status.error_message();
}
