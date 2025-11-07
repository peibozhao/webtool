#include "map_pinning.h"
#include "google/protobuf/util/json_util.h"
#include "spdlog/spdlog.h"
#include "utils/log.h"
#include "utils/utils.h"

MapPinning::MapPinning() {
  std::filesystem::path sqlite_db_fpath =
      GetDataDirectory() / (GetProcessName() + ".db");
  SPDLOG_INFO("Open sqlite database file {}", sqlite_db_fpath.string());
  SQLITE_ASSERT(sqlite3_open(sqlite_db_fpath.c_str(), &sqlite_handle_));

  const char *create_sql = R"(
      CREATE TABLE IF NOT EXISTS map_pinning (
          key TEXT PRIMARY KEY,
          password TEXT,
          value TEXT
      );
  )";
  SQLITE_ASSERT(
      sqlite3_exec(sqlite_handle_, create_sql, nullptr, nullptr, nullptr));

  const char *insert_sql = R"(
      INSERT OR REPLACE INTO map_pinning (key, password, value)
      SELECT ?, ?, ?
      WHERE NOT EXISTS (SELECT 1 FROM map_pinning WHERE key = ? AND password <> ?);
      )";

  SQLITE_ASSERT(sqlite3_prepare_v2(sqlite_handle_, insert_sql, -1,
                                   &sqlite_insert_statement_, nullptr));

  const char *search_sql = R"(SELECT value FROM map_pinning WHERE key = ?;)";
  SQLITE_ASSERT(sqlite3_prepare_v2(sqlite_handle_, search_sql, -1,
                                   &sqlite_search_statement_, nullptr));
}

grpc::Status MapPinning::Upload(grpc::ServerContext *context,
                                const UploadRequest *request,
                                UploadResponse *response) {
  std::string save_text;
  absl::Status absl_status =
      google::protobuf::util::MessageToJsonString(*request, &save_text);
  if (!absl_status.ok()) {
    SPDLOG_ERROR("Convert protobuf to json failed. code={}",
                 int(absl_status.code()));
    return grpc::Status(grpc::StatusCode::INTERNAL, "格式转换错误");
  }

  SPDLOG_INFO("Save map marker set {} password {} text {}", request->name(),
              request->password(), save_text);

  if (request->name().empty()) {
    SPDLOG_ERROR("Map marker set name is empty");
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "集合名称不能为空");
  }
  SQLITE_ASSERT(sqlite3_reset(sqlite_insert_statement_));
  SQLITE_ASSERT(sqlite3_bind_text(sqlite_insert_statement_, 1,
                                  request->name().c_str(), -1,
                                  SQLITE_TRANSIENT));
  SQLITE_ASSERT(sqlite3_bind_text(sqlite_insert_statement_, 2,
                                  request->password().c_str(), -1,
                                  SQLITE_TRANSIENT));
  SQLITE_ASSERT(sqlite3_bind_text(sqlite_insert_statement_, 3,
                                  save_text.c_str(), -1, SQLITE_TRANSIENT));
  SQLITE_ASSERT(sqlite3_bind_text(sqlite_insert_statement_, 4,
                                  request->name().c_str(), -1,
                                  SQLITE_TRANSIENT));
  SQLITE_ASSERT(sqlite3_bind_text(sqlite_insert_statement_, 5,
                                  request->password().c_str(), -1,
                                  SQLITE_TRANSIENT));

  int sqlite_ret = sqlite3_step(sqlite_insert_statement_);
  if (sqlite_ret != SQLITE_DONE) {
    SPDLOG_ERROR("Sqlite insert failed. code={}", sqlite_ret);
    return grpc::Status(grpc::StatusCode::INTERNAL, "数据存储异常");
  }

  sqlite_ret = sqlite3_changes(sqlite_handle_);
  if (sqlite_ret == 0) {
    SPDLOG_WARN("Map markers insert failed, password error");
    return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "密码错误");
  } else if (sqlite_ret < 0) {
    SPDLOG_ERROR("Map markers insert failed. code={}", sqlite_ret);
    return grpc::Status(grpc::StatusCode::INTERNAL, "数据存储异常");
  }

  SQLITE_ASSERT(sqlite3_reset(sqlite_insert_statement_));
  return grpc::Status::OK;
}

grpc::Status MapPinning::Download(grpc::ServerContext *context,
                                  const DownloadRequest *request,
                                  DownloadResponse *response) {
  if (request->name().empty()) {
    SPDLOG_ERROR("Map marker set name is empty");
    return grpc::Status::OK;
  }
  SQLITE_ASSERT(sqlite3_reset(sqlite_search_statement_));
  SQLITE_ASSERT(sqlite3_bind_text(sqlite_search_statement_, 1,
                                  request->name().c_str(), -1,
                                  SQLITE_TRANSIENT));

  int sqlite_ret = sqlite3_step(sqlite_search_statement_);
  if (sqlite_ret == SQLITE_DONE) {
    SPDLOG_WARN("Map marker set {} not found", request->name());
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "地点集合不存在");
  } else if (sqlite_ret != SQLITE_ROW) {
    SPDLOG_ERROR("Sqlite search failed. code={}", sqlite_ret);
    return grpc::Status(grpc::StatusCode::INTERNAL, "数据查找异常");
  }

  std::string text =
      (const char *)sqlite3_column_text(sqlite_search_statement_, 0);

  SPDLOG_INFO("Get map marker set {} text '{}'", request->name(), text);
  google::protobuf::json::ParseOptions parse_options;
  parse_options.ignore_unknown_fields = true;
  absl::Status absl_status = google::protobuf::util::JsonStringToMessage(
      text, response, parse_options);

  if (!absl_status.ok()) {
    SPDLOG_ERROR("Parse json to protobuf failed. code={}, text='{}'",
                 int(absl_status.code()), text);
    return grpc::Status(grpc::StatusCode::INTERNAL, "数据转换异常");
  }
  SQLITE_ASSERT(sqlite3_reset(sqlite_search_statement_));
  return grpc::Status::OK;
}
