
#include "qr_code.grpc.pb.h"

class QrCode : public QrCodeService::Service {
public:
  grpc::Status GenerateImage(grpc::ServerContext *context,
                             const ::GenerateImageRequest *request,
                             GenerateImageResponse *response) override;
  grpc::Status ParseText(grpc::ServerContext *context,
                         const ::ParseTextRequest *request,
                         ParseTextResponse *response) override;
};

