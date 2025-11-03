#pragma once

#include "map_pinning.grpc.pb.h"
#include "sqlite3.h"

class MapPinning : public MapPinningService::Service {
private:
  struct Point {
    float lat, lon;
    std::string text;
  };

public:
  MapPinning();

  grpc::Status Upload(grpc::ServerContext *context,
                      const UploadRequest *request,
                      UploadResponse *response) override;

  grpc::Status Download(grpc::ServerContext *context,
                        const DownloadRequest *request,
                        DownloadResponse *response) override;

private:
  sqlite3 *sqlite_handle_ = nullptr;
  sqlite3_stmt *sqlite_insert_statement_ = nullptr;
  sqlite3_stmt *sqlite_search_statement_ = nullptr;
};
