#include "copy_text.h"
#include "spdlog/spdlog.h"
#include "sqlite3.h"
#include "utils/log.h"
#include "utils/utils.h"

CopyText::CopyText() {
  std::filesystem::path sqlite_db_fpath =
      GetDataDirectory() / (GetProcessName() + ".db");
  SPDLOG_INFO("Open sqlite database file {}", sqlite_db_fpath.string());
  SQLITE_ASSERT(sqlite3_open(sqlite_db_fpath.c_str(), &sqlite_handle_));

  const char *create_sql = R"(
      CREATE TABLE IF NOT EXISTS copy (
          key TEXT PRIMARY KEY,
          value TEXT,
          timestamp INTEGER DEFAULT (strftime('%s','now'))
      );
  )";
  SQLITE_ASSERT(
      sqlite3_exec(sqlite_handle_, create_sql, nullptr, nullptr, nullptr));

  const char *trigger_sql = R"(
      CREATE TRIGGER IF NOT EXISTS delete_expired
      BEFORE INSERT ON copy
      BEGIN
        DELETE FROM copy WHERE timestamp <= strftime('%s', 'now') - 86400;
      END;)";
  SQLITE_ASSERT(
      sqlite3_exec(sqlite_handle_, trigger_sql, nullptr, nullptr, nullptr));

  const char *insert_sql = R"(
      INSERT INTO copy (key, value)
      VALUES (lower(hex(randomblob(1))), ?)
      RETURNING key;)";
  SQLITE_ASSERT(sqlite3_prepare_v2(sqlite_handle_, insert_sql, -1,
                                   &sqlite_insert_statement_, nullptr));

  const char *search_sql = R"(SELECT value FROM copy WHERE key = ?;)";
  SQLITE_ASSERT(sqlite3_prepare_v2(sqlite_handle_, search_sql, -1,
                                   &sqlite_search_statement_, nullptr));
}

grpc::Status CopyText::Submit(grpc::ServerContext *context,
                              const SubmitRequest *request,
                              SubmitResponse *response) {
  if (request->text().empty()) {
    SPDLOG_WARN("Copy submit text is empty, skip");
    return grpc::Status::OK;
  }
  SQLITE_ASSERT(sqlite3_reset(sqlite_insert_statement_));
  SQLITE_ASSERT(sqlite3_bind_text(sqlite_insert_statement_, 1,
                                  request->text().c_str(), -1,
                                  SQLITE_TRANSIENT));
  if (sqlite3_step(sqlite_insert_statement_) != SQLITE_ROW) {
    return grpc::Status(grpc::StatusCode::INTERNAL, "Sqlite insert failed");
  }
  std::string code =
      (const char *)sqlite3_column_text(sqlite_insert_statement_, 0);
  SPDLOG_INFO("Save text '{}' with code {}", request->text(), code);

  response->set_code(code);
  code_text_map_[response->code()] = request->text();
  current_code_ += 1;

  SQLITE_ASSERT(sqlite3_reset(sqlite_insert_statement_));
  return grpc::Status::OK;
}

grpc::Status CopyText::Retrieve(grpc::ServerContext *context,
                                const RetrieveRequest *request,
                                RetrieveResponse *response) {
  SQLITE_ASSERT(sqlite3_reset(sqlite_search_statement_));
  SQLITE_ASSERT(sqlite3_bind_text(sqlite_search_statement_, 1,
                                  request->code().c_str(), -1,
                                  SQLITE_TRANSIENT));

  int step_ret = sqlite3_step(sqlite_search_statement_);
  if (step_ret == SQLITE_DONE) {
    SPDLOG_WARN("Copy code {} not found", request->code());
    return grpc::Status::OK;
  } else if (step_ret != SQLITE_ROW) {
    return grpc::Status(
        grpc::StatusCode::INTERNAL,
        std::format("Sqlite search failed. code={}", request->code()));
  }

  std::string text =
      (const char *)sqlite3_column_text(sqlite_search_statement_, 0);
  response->set_text(text);
  SPDLOG_INFO("Return text '{}' with code {}", response->text(),
              request->code());
  SQLITE_ASSERT(sqlite3_reset(sqlite_search_statement_));
  return grpc::Status::OK;
}
