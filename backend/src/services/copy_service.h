#pragma once

#include "../service.h"

class CopyService : public Service {
public:
  CopyService(httplib::Server &http_server);

  void Submit(const std::string &text, std::string &code);

  void Retrieve(const std::string &code, std::string &text);

private:
  static constexpr int max_code_ = 99;
  const int code_digits_ = std::to_string(std::abs(max_code_)).length();
  int current_code_ = 0;
  std::unordered_map<std::string, std::string> code_text_map_;
};

