#pragma once

#include "copy_service.pb.h"
#include "google/protobuf/message.h"
#include "grpcpp/support/status.h"
#include "httplib.h"
#include "map_pinning.pb.h"
#include "nlohmann/json.hpp"
#include "qr_code.pb.h"
#include "spdlog/spdlog.h"
#include "super_resolution_service.pb.h"

template <typename M>
concept DerivedFromMessage = std::derived_from<M, google::protobuf::Message>;

template <DerivedFromMessage M>
void ConvertHTTPRequestToProto(const httplib::Request &req, M *msg) {
  SPDLOG_INFO("Use empty HTTP request to GRPC request");
}

template <DerivedFromMessage M>
void ConvertProtoToHTTPResponse(const M &msg, httplib::Response *res) {
  SPDLOG_INFO("Use empty GRPC response to HTTP response");
}

void ConvertGRPCStatusToHTTPCode(const grpc::Status &grpc_status,
                                 httplib::Response &res);

template <>
inline void ConvertHTTPRequestToProto(const httplib::Request &req,
                                      SubmitRequest *msg) {
  msg->set_text(req.get_param_value("text"));
}

template <>
inline void ConvertProtoToHTTPResponse(const SubmitResponse &msg,
                                       httplib::Response *res) {
  res->set_content(std::format(R"({{"code": "{}"}})", msg.code()),
                   "application/json");
}

template <>
inline void ConvertHTTPRequestToProto(const httplib::Request &req,
                                      RetrieveRequest *msg) {
  msg->set_code(req.get_param_value("code"));
}

template <>
inline void ConvertProtoToHTTPResponse(const RetrieveResponse &msg,
                                       httplib::Response *res) {
  res->set_content(std::format(R"({{"text": "{}"}})", msg.text()),
                   "application/json");
}

template <>
inline void ConvertHTTPRequestToProto(const httplib::Request &req,
                                      GenerateImageRequest *msg) {
  msg->set_text(req.get_param_value("text"));
}

template <>
inline void ConvertProtoToHTTPResponse(const GenerateImageResponse &msg,
                                       httplib::Response *res) {
  res->set_content(msg.image(), "image/jpg");
}

template <>
inline void ConvertHTTPRequestToProto(const httplib::Request &req,
                                      ParseTextRequest *msg) {
  const httplib::FormData file_form_data = req.form.get_file("image");
  if (file_form_data.content.empty()) {
    throw std::invalid_argument("File content is empty");
  }
  msg->set_image(file_form_data.content);
}

template <>
inline void ConvertProtoToHTTPResponse(const ParseTextResponse &msg,
                                       httplib::Response *res) {
  res->set_content(std::format(R"({{"text":"{}"}})", msg.text()),
                   "application/json");
}

template <>
inline void ConvertHTTPRequestToProto(const httplib::Request &req,
                                      ImageRequest *msg) {
  const httplib::FormData file_form_data = req.form.get_file("image");
  if (file_form_data.content.empty()) {
    throw std::invalid_argument("File content is empty");
  }
  msg->set_image(file_form_data.content);
}

template <>
inline void ConvertProtoToHTTPResponse(const ImageResponse &msg,
                                       httplib::Response *res) {
  res->set_content(msg.image(), "image/jpg");
}

template <>
inline void ConvertHTTPRequestToProto(const httplib::Request &req,
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
inline void ConvertHTTPRequestToProto(const httplib::Request &req,
                                      DownloadRequest *msg) {
  msg->set_name(req.get_param_value("name"));
}

template <>
inline void ConvertProtoToHTTPResponse(const DownloadResponse &msg,
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
