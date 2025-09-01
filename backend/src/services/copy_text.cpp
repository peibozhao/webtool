
#include "copy_text.h"
#include "spdlog/spdlog.h"

grpc::Status CopyText::Submit(grpc::ServerContext *context,
                              const SubmitRequest *request,
                              SubmitResponse *response) {
  response->set_code(std::format("{:0{}}", current_code_, code_digits_));
  code_text_map_[response->code()] = request->text();
  current_code_ += 1;
  SPDLOG_INFO("Save text {} with code {}", request->text(), response->code());
  return grpc::Status::OK;
}

grpc::Status CopyText::Retrieve(grpc::ServerContext *context,
                                const RetrieveRequest *request,
                                RetrieveResponse *response) {
  int code_num = -1;
  auto iter = code_text_map_.find(request->code());
  if (iter == code_text_map_.end()) {
    SPDLOG_INFO("Code {} is not exist", request->code());
  } else {
    response->set_text(iter->second);
    SPDLOG_INFO("Return text {} with code {}", response->text(),
                request->code());
  }
  return grpc::Status::OK;
}
