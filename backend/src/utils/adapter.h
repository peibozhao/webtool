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

void ConvertGRPCStatusToHTTPCode(const grpc::Status &grpc_status,
                                 httplib::Response &res);

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

#define DEFINE_GRPC_HTTP_CONVERT(rpc, request_method, response_method)         \
  template <>                                                                  \
  inline void ConvertHTTPRequestToProto(const httplib::Request &req,           \
                                        rpc##Request *msg) {                   \
    request_method;                                                            \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline void ConvertProtoToHTTPResponse(const rpc##Response &msg,             \
                                         httplib::Response *res) {             \
    response_method;                                                           \
  }

DEFINE_GRPC_HTTP_CONVERT(Submit,
    msg->set_text(req.get_param_value("text")),
    res->set_content(std::format(R"({{"code": "{}"}})",
        msg.code()), "application/json"));

DEFINE_GRPC_HTTP_CONVERT(Retrieve,
    msg->set_code(req.get_param_value("code")),
    res->set_content(std::format(R"({{"text": "{}"}})",
        msg.text()), "application/json"));
 
DEFINE_GRPC_HTTP_CONVERT(GenerateImage,
  msg->set_text(req.get_param_value("text")),
  res->set_content(msg.image(), "image/jpg"));

DEFINE_GRPC_HTTP_CONVERT(
    ParseText,
    const httplib::FormData file_form_data = req.form.get_file("image");
    if (file_form_data.content.empty()) {
      throw std::invalid_argument("File content is empty");
    }
    msg->set_image(file_form_data.content);,
    res->set_content(std::format(R"({{"text":"{}"}})", msg.text()),
                       "application/json"));

DEFINE_GRPC_HTTP_CONVERT(Times4,
  const httplib::FormData file_form_data = req.form.get_file("image");
  if (file_form_data.content.empty()) {
    throw std::invalid_argument("File content is empty");
  }
  msg->set_image(file_form_data.content);,
  res->set_content(msg.image(), "image/jpg"));

DEFINE_GRPC_HTTP_CONVERT(Download, 
  msg->set_name(req.get_param_value("name")),
  nlohmann::json res_json;
  for (const Marker &marker : msg.markers()) {
    nlohmann::json marker_json;
    marker_json["lat"] = marker.lat();
    marker_json["lng"] = marker.lng();
    marker_json["text"] = marker.text();
    res_json["markers"].push_back(marker_json);
  }
  res->body = res_json.dump();)

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

