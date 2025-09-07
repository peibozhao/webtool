
cmake_minimum_required(VERSION 3.16)
project(webtool)
set(CMAKE_CXX_STANDARD 20)

include(ExternalProject)
include(FetchContent)

FetchContent_Declare(gflags
  GIT_REPOSITORY https://github.com/gflags/gflags
  GIT_TAG v2.2.2)

FetchContent_Declare(cpp-httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib
  GIT_TAG v0.26.0)

FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.x)

FetchContent_Declare(grpc
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG v1.74.1)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(FETCHCONTENT_QUIET OFF)

FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include/)

FetchContent_MakeAvailable(cpp-httplib)
include_directories(${cpp-httplib_SOURCE_DIR})

FetchContent_MakeAvailable(spdlog)
add_definitions(-DSPDLOG_USE_STD_FORMAT)
include_directories(${spdlog_SOURCE_DIR}/include)

# gRPC
set(ABSL_ENABLE_INSTALL ON)  # Avoid CMake errors in gRPC
set(gRPC_BUILD_TESTS OFF)
set(gRPC_BUILD_CSHARP_EXT OFF)
set(gRPC_BUILD_GRPC_CSHARP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_NODE_PLUGIN OFF)
set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PHP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PYTHON_PLUGIN OFF)
set(gRPC_BUILD_GRPC_RUBY_PLUGIN OFF)
FetchContent_MakeAvailable(grpc)
set(PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(GRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)
include_directories(${protobuf_SOURCE_DIR}/src/)
include_directories(${absl_SOURCE_DIR})

# Compile proto file
set(PROTO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
message(STATUS "Proto source directory ${PROTO_SOURCE_DIR}")
set(PROTO_GENERATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
message(STATUS "Proto generate directory ${PROTO_GENERATE_DIR}")
file(MAKE_DIRECTORY ${PROTO_GENERATE_DIR})

file(GLOB PROTO_SOURCE_FILES "${PROTO_SOURCE_DIR}/*.proto")
foreach(PROTO_SOURCE_FILE ${PROTO_SOURCE_FILES})
  message(STATUS "Compile proto file ${PROTO_SOURCE_FILE}")
  get_filename_component(PROTO_SOURCE_FNAME_WE ${PROTO_SOURCE_FILE} NAME_WE)
  get_filename_component(PROTO_SOURCE_FNAME ${PROTO_SOURCE_FILE} NAME)
  set(PROTO_GENERATE_CPP_FILES)
  list(APPEND PROTO_GENERATE_CPP_FILES
    "${PROTO_GENERATE_DIR}/${PROTO_SOURCE_FNAME_WE}.pb.h"
    "${PROTO_GENERATE_DIR}/${PROTO_SOURCE_FNAME_WE}.pb.cc"
    "${PROTO_GENERATE_DIR}/${PROTO_SOURCE_FNAME_WE}.grpc.pb.h"
    "${PROTO_GENERATE_DIR}/${PROTO_SOURCE_FNAME_WE}.grpc.pb.cc")
  message(STATUS "Proto file all generate ${PROTO_GENERATE_CPP_FILES}")
  add_custom_command(
    OUTPUT ${PROTO_GENERATE_CPP_FILES}
    COMMAND ${PROTOBUF_PROTOC}
    ARGS --grpc_out ${PROTO_GENERATE_DIR}
      --cpp_out ${PROTO_GENERATE_DIR}
      -I ${PROTO_SOURCE_DIR}
      --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}"
      ${PROTO_SOURCE_FNAME}
      DEPENDS "${PROTO_SOURCE_DIR}/${PROTO_SOURCE_FNAME}")
    list(APPEND PROTO_ALL_GENERATE_CPP_FILES ${PROTO_GENERATE_CPP_FILES})
endforeach(PROTO_SOURCE_FILE)

include_directories(${CMAKE_BINARY_DIR}/gens)  # gRPC proto generated
include_directories(${PROTO_GENERATE_DIR})

# opencv
set(opencv_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/opencv-src)
set(opencv_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/opencv-build)
set(opencv_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/opencv-install)

ExternalProject_Add(
  opencv
  GIT_REPOSITORY https://github.com/opencv/opencv
  GIT_TAG 4.12.0
  UPDATE_DISCONNECTED ON
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${opencv_INSTALL_DIR}
  SOURCE_DIR ${opencv_SOURCE_DIR}
  BINARY_DIR ${opencv_BINARY_DIR}
  BUILD_ALWAYS OFF
)

find_package(opencv PATHS ${opencv_INSTALL_DIR})

include_directories(${opencv_INCLUDE_DIRS})

# Compile
file(GLOB_RECURSE BACKEND_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(backend ${BACKEND_SOURCES} ${PROTO_ALL_GENERATE_CPP_FILES})
target_link_libraries(backend grpc++ gflags ${OpenCV_LIBRARIES})

