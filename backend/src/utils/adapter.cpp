#include "adapter.h"
#include "nlohmann/json.hpp"

void ConvertGRPCStatusToHTTPCode(const grpc::Status &grpc_status,
                                 httplib::Response &res) {
  if (grpc_status.ok()) {
    return;
  }
  SPDLOG_ERROR("GRPC call return failed. code={}, message='{}'",
               int(grpc_status.error_code()), grpc_status.error_message());
  int grpc_errcode = grpc_status.error_code();
  switch (grpc_status.error_code()) {
  case grpc::StatusCode::UNAUTHENTICATED:
    res.status = 403;
    res.body = "密码错误";
    break;
  case grpc::StatusCode::NOT_FOUND:
    res.status = 404;
    res.body = "资源不存在";
    break;
  default:
    res.status = 500;
    res.body = "未知错误";
    break;
  }
}

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               SubmitRequest *msg) {
  msg->set_text(req.get_param_value("text"));
}

template <>
void ConvertProtoToHTTPResponse(const SubmitResponse &msg,
                                httplib::Response *res) {
  res->set_content(std::format(R"({{"code": "{}"}})", msg.code()),
                   "application/json");
}

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               RetrieveRequest *msg) {
  msg->set_code(req.get_param_value("code"));
}

template <>
void ConvertProtoToHTTPResponse(const RetrieveResponse &msg,
                                httplib::Response *res) {
  res->set_content(std::format(R"({{"text": "{}"}})", msg.text()),
                   "application/json");
}

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               GenerateImageRequest *msg) {
  msg->set_text(req.get_param_value("text"));
}

template <>
void ConvertProtoToHTTPResponse(const GenerateImageResponse &msg,
                                httplib::Response *res) {
  res->set_content(msg.image(), "image/jpg");
}

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               ParseTextRequest *msg) {
  const httplib::FormData file_form_data = req.form.get_file("image");
  if (file_form_data.content.empty()) {
    throw std::invalid_argument("File content is empty");
  }
  msg->set_image(file_form_data.content);
}

template <>
void ConvertProtoToHTTPResponse(const ParseTextResponse &msg,
                                httplib::Response *res) {
  res->set_content(std::format(R"({{"text":"{}"}})", msg.text()),
                   "application/json");
}

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req, ImageRequest *msg) {
  const httplib::FormData file_form_data = req.form.get_file("image");
  if (file_form_data.content.empty()) {
    throw std::invalid_argument("File content is empty");
  }
  msg->set_image(file_form_data.content);
}

template <>
void ConvertProtoToHTTPResponse(const ImageResponse &msg,
                                httplib::Response *res) {
  res->set_content(msg.image(), "image/jpg");
}

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               UploadRequest *msg) {
  nlohmann::json req_json = nlohmann::json::parse(req.body);
  if (req_json["name"].get<std::string>().empty()) {
    throw std::invalid_argument("Upload map markers set name is empty");
  }
  msg->set_name(req_json["name"].get<std::string>());
  msg->set_password(req_json["password"].get<std::string>());
  for (const nlohmann::json &marker_json : req_json["markers"]) {
    Marker *marker_pb = msg->add_markers();
    marker_pb->set_lat(marker_json["lat"].get<float>());
    marker_pb->set_lng(marker_json["lng"].get<float>());
    marker_pb->set_text(marker_json["text"].get<std::string>());
  }
}

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               DownloadRequest *msg) {
  msg->set_name(req.get_param_value("name"));
}

template <>
void ConvertProtoToHTTPResponse(const DownloadResponse &msg,
                                httplib::Response *res) {
  nlohmann::json res_json;
  for (const Marker &marker : msg.markers()) {
    nlohmann::json marker_json;
    marker_json["lat"] = marker.lat();
    marker_json["lng"] = marker.lng();
    marker_json["text"] = marker.text();
    res_json["markers"].push_back(marker_json);
  }
  res->body = res_json.dump();
}
