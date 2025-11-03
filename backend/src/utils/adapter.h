#pragma once

#include "copy_service.pb.h"
#include "google/protobuf/message.h"
#include "grpcpp/support/status.h"
#include "httplib.h"
#include "map_pinning.pb.h"
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

// copy
template <>
void ConvertHTTPRequestToProto(const httplib::Request &req, SubmitRequest *msg);

template <>
void ConvertProtoToHTTPResponse(const SubmitResponse &msg,
                                httplib::Response *res);

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               RetrieveRequest *msg);

template <>
void ConvertProtoToHTTPResponse(const RetrieveResponse &msg,
                                httplib::Response *res);

// QR code
template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               GenerateImageRequest *msg);

template <>
void ConvertProtoToHTTPResponse(const GenerateImageResponse &msg,
                                httplib::Response *res);

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               ParseTextRequest *msg);

template <>
void ConvertProtoToHTTPResponse(const ParseTextResponse &msg,
                                httplib::Response *res);

// super resolution
template <>
void ConvertHTTPRequestToProto(const httplib::Request &req, ImageRequest *msg);

template <>
void ConvertProtoToHTTPResponse(const ImageResponse &msg,
                                httplib::Response *res);

// map pinning
template <>
void ConvertHTTPRequestToProto(const httplib::Request &req, UploadRequest *msg);

template <>
void ConvertHTTPRequestToProto(const httplib::Request &req,
                               DownloadRequest *msg);

template <>
void ConvertProtoToHTTPResponse(const DownloadResponse &msg,
                                httplib::Response *res);
