#pragma once

#include "copy_service.grpc.pb.h"
#include "copy_service.pb.h"
#include "sqlite3.h"

class CopyText : public CopyService::Service {
public:
  CopyText();

  grpc::Status Submit(grpc::ServerContext *context,
                      const SubmitRequest *request,
                      SubmitResponse *response) override;

  grpc::Status Retrieve(grpc::ServerContext *context,
                        const RetrieveRequest *request,
                        RetrieveResponse *response) override;

private:
  sqlite3 *sqlite_handle_ = nullptr;
  sqlite3_stmt *sqlite_insert_statement_ = nullptr;
  sqlite3_stmt *sqlite_search_statement_ = nullptr;

  static constexpr int max_code_ = 99;
  const int code_digits_ = std::to_string(std::abs(max_code_)).length();
  int current_code_ = 0;
  std::unordered_map<std::string, std::string> code_text_map_;
};

