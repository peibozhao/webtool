#pragma once

#include "httplib.h"
#include <format>
#include <unordered_map>

#define ASSERT(expression)                                                     \
  {                                                                            \
    bool status = expression;                                                  \
    if (!status) {                                                             \
      SPDLOG_ERROR("Assert {} failed", #expression);                           \
      abort();                                                                 \
    }                                                                          \
  }

#define CUDA_ASSERT(expression)                                                \
  {                                                                            \
    cudaError status = expression;                                             \
    if (status != cudaSuccess) {                                               \
      SPDLOG_ERROR("cuda function return failed, status is {}", int(status));  \
      abort();                                                                 \
    }                                                                          \
  }

#define NVJPEG_ASSERT(expression)                                              \
  {                                                                            \
    nvjpegStatus_t status = expression;                                        \
    if (status != NVJPEG_STATUS_SUCCESS) {                                     \
      SPDLOG_ERROR("nvjpeg function return failed, status is {}",              \
                   int(status));                                               \
      abort();                                                                 \
    }                                                                          \
  }

#define SQLITE_ASSERT(expression)                                              \
  {                                                                            \
    int status = expression;                                                   \
    if (status != SQLITE_OK) {                                                 \
      SPDLOG_ERROR("sqlite function return failed, status is {}",              \
                   int(status));                                               \
      abort();                                                                 \
    }                                                                          \
  }

template <>
struct std::formatter<std::unordered_map<std::string, std::string>> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const std::unordered_map<std::string, std::string> &path_params,
              format_context &fc) const {
    if (path_params.empty()) {
      return fc.out();
    }
    auto iter = path_params.begin();
    for (int i = 0; i < path_params.size() - 1; ++i) {
      std::format_to(fc.out(), "{}: {}, ", iter->first, iter->second);
      ++iter;
    }
    return std::format_to(fc.out(), "{}: {}", iter->first, iter->second);
  }
};

template <> struct std::formatter<httplib::Params> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Params &params, format_context &fc) const {
    if (params.empty()) {
      return fc.out();
    }
    std::for_each_n(params.begin(), params.size() - 1,
                    [&fc](const std::pair<std::string, std::string> &param) {
                      std::format_to(fc.out(), "{}: {}, ", param.first,
                                     param.second);
                    });
    return std::format_to(fc.out(), "{}: {}", params.rbegin()->first,
                          params.rbegin()->second);
  }
};

template <> struct std::formatter<httplib::Headers> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Headers &headers, format_context &fc) const {
    if (headers.empty()) {
      return fc.out();
    }
    auto iter = headers.begin();
    for (int i = 0; i < headers.size() - 1; ++i) {
      std::format_to(fc.out(), "{}: {}, ", iter->first, iter->second);
      ++iter;
    }
    return std::format_to(fc.out(), "{}: {}", iter->first, iter->second);
  }
};

template <> struct std::formatter<httplib::Request> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Request &request, format_context &fc) const {
    return std::format_to(fc.out(),
                          "address({}:{}), method({}), path({}), params({}), "
                          "headers({})",
                          request.remote_addr, request.remote_port,
                          request.method, request.path, request.params,
                          request.headers);
  }
};

template <> struct std::formatter<httplib::Response> {
  constexpr auto parse(format_parse_context &fpc) { return fpc.begin(); }

  auto format(const httplib::Response &response, format_context &fc) const {
    return std::format_to(fc.out(), "staus({}), header({})", response.status,
                          response.headers);
  }
};
