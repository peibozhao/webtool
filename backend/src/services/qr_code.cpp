
#include "qr_code.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "spdlog/spdlog.h"

grpc::Status QrCode::GenerateImage(grpc::ServerContext *context,
                                   const ::GenerateImageRequest *request,
                                   GenerateImageResponse *response) {
  cv::QRCodeEncoder::Params qr_encoder_params;
  std::shared_ptr<cv::QRCodeEncoder> qr_encoder =
      cv::QRCodeEncoder::create(qr_encoder_params);
  cv::Mat image;
  qr_encoder->encode(request->text(), image);
  cv::resize(image, image, cv::Size(500, 500), 0, 0, cv::INTER_NEAREST);
  std::vector<uchar> encode_buffer;
  if (!cv::imencode(".jpg", image, encode_buffer)) {
    SPDLOG_ERROR("Image encode failed");
    return grpc::Status(grpc::StatusCode::UNKNOWN, "Image encode error");
  }
  response->set_image(encode_buffer.data(), encode_buffer.size());
  return grpc::Status::OK;
}
grpc::Status QrCode::ParseText(grpc::ServerContext *context,
                               const ::ParseTextRequest *request,
                               ParseTextResponse *response) {
  cv::QRCodeDetector qrcode_detector;
  std::vector<char> buffer(request->image().begin(), request->image().end());
  cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
  std::string text = qrcode_detector.detectAndDecode(image);
  SPDLOG_INFO("Parse QR code image, text is {}", text);
  response->set_text(text);
  return grpc::Status::OK;
}
