
#include "copy_service.h"
#include "spdlog/spdlog.h"

CopyService::CopyService(httplib::Server &http_server) : Service(http_server) {
  SPDLOG_INFO("Set copy service handler");
  http_server.Post("/api/copy/submit",
                   [this](const httplib::Request &req, httplib::Response &res) {
                     std::string code;
                     this->Submit(req.get_param_value("text"), code);
                     res.set_content(std::format(R"({{"code": "{}"}})", code),
                                     "application/json");
                   });
  http_server.Get("/api/copy/retrieve",
                  [this](const httplib::Request &req, httplib::Response &res) {
                    std::string text;
                    this->Retrieve(req.get_param_value("code"), text);
                    res.set_content(std::format(R"({{"text": "{}"}})", text),
                                    "application/json");
                  });
}

void CopyService::Submit(const std::string &text, std::string &code) {
  code = std::format("{:0{}}", current_code_, code_digits_);
  code_text_map_[code] = text;
  current_code_ += 1;
  SPDLOG_INFO("Save text {} with code {}", text, code);
}

void CopyService::Retrieve(const std::string &code, std::string &text) {
  int code_num = -1;
  auto iter = code_text_map_.find(code);
  if (iter == code_text_map_.end()) {
    SPDLOG_INFO("Code {} is not exist", code);
  } else {
    text = iter->second;
    SPDLOG_INFO("Return text {} with code {}", text, code);
  }
}
