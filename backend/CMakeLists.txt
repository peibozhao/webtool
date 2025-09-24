
cmake_minimum_required(VERSION 3.16)

project(backend LANGUAGES CXX)

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

FetchContent_Declare(opencv
  GIT_REPOSITORY https://github.com/opencv/opencv
  GIT_TAG 4.12.0)

FetchContent_Declare(sqlite
  GIT_REPOSITORY https://github.com/sqlite/sqlite
  GIT_TAG version-3.42.1)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(FETCHCONTENT_QUIET OFF)

# gflags
FetchContent_MakeAvailable(gflags)

# cpp-httplib
FetchContent_MakeAvailable(cpp-httplib)

# spdlog
FetchContent_MakeAvailable(spdlog)

# gRPC
set(ABSL_ENABLE_INSTALL ON)  # Avoid CMake errors in gRPC
set(gRPC_BUILD_TESTS OFF)
set(gRPC_BUILD_GRPC_CSHARP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_NODE_PLUGIN OFF)
set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PHP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PYTHON_PLUGIN OFF)
set(gRPC_BUILD_GRPC_RUBY_PLUGIN OFF)
FetchContent_MakeAvailable(grpc)
set(PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(GRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)

# opencv
set(opencv_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/opencv-install)
FetchContent_Populate(opencv)
ExternalProject_Add(
  opencv
  UPDATE_DISCONNECTED ON
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${opencv_INSTALL_DIR}
    -DBUILD_TESTS=OFF
    -DBUILD_PERF_TESTS=OFF
    -DBUILD_opencv_apps=OFF
  SOURCE_DIR ${opencv_SOURCE_DIR}
  BINARY_DIR ${opencv_BINARY_DIR}
  BUILD_ALWAYS OFF
)

# sqlite
set(sqlite_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/sqlite-install)
FetchContent_Populate(sqlite)
ExternalProject_Add(
  sqlite
  UPDATE_DISCONNECTED ON
  CONFIGURE_COMMAND ${sqlite_SOURCE_DIR}/configure --prefix=${sqlite_INSTALL_DIR} --disable-shared --enable-static
  SOURCE_DIR ${sqlite_SOURCE_DIR}
  BINARY_DIR ${sqlite_BINARY_DIR}
  BUILD_COMMAND make
  BUILD_ALWAYS OFF
)

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

# third party
include_directories(${gflags_BINARY_DIR}/include/)
include_directories(${cpp-httplib_SOURCE_DIR})
add_definitions(-DSPDLOG_USE_STD_FORMAT)
include_directories(${spdlog_SOURCE_DIR}/include)
include_directories(${protobuf_SOURCE_DIR}/src/)
include_directories(${absl_SOURCE_DIR})
include_directories(${opencv_INSTALL_DIR}/include/opencv4)
include_directories(${sqlite_INSTALL_DIR}/include)
link_directories(${opencv_INSTALL_DIR}/lib)
link_directories(${sqlite_INSTALL_DIR}/lib)
set(opencv_LIBRARIES opencv_imgproc opencv_objdetect opencv_imgcodecs opencv_core)

# proto generation header
include_directories(${CMAKE_BINARY_DIR}/gens)  # gRPC proto generated
include_directories(${PROTO_GENERATE_DIR})

# cuda
set(CMAKE_CXX_STANDARD 23)
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  option(USE_CUDA "Build with CUDA support" ON)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  enable_language(CUDA)
else()
  message(STATUS "CUDA Toolkit not found, building without CUDA support.")
  option(USE_CUDA "Build with CUDA support" OFF)
endif()

# Compile
file(GLOB_RECURSE BACKEND_CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(CMAKE_INSTALL_RPATH \$ORIGIN/../lib)
if (USE_CUDA)
  file(GLOB_RECURSE BACKEND_CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
  add_definitions(-DUSE_CUDA)
  include_directories(SYSTEM /usr/local/cuda/include)
  include_directories(SYSTEM /usr/local/tensorrt/include)
  set(CUDA_LIBRARIES CUDA::cudart CUDA::nvjpeg nvinfer)
  add_executable(backend ${BACKEND_CPP_SOURCES} ${PROTO_ALL_GENERATE_CPP_FILES} ${BACKEND_CUDA_SOURCES})
  target_link_libraries(backend grpc++_reflection grpc++ gflags ${opencv_LIBRARIES} sqlite3 ${CUDA_LIBRARIES})
else()
  add_executable(backend ${BACKEND_CPP_SOURCES} ${PROTO_ALL_GENERATE_CPP_FILES})
  target_link_libraries(backend grpc++_reflection grpc++ gflags ${opencv_LIBRARIES} sqlite3)
endif()
add_dependencies(backend opencv)

# Install
if (${CMAKE_SYSTEM_NAME} STREQUAL Linux)
  install(TARGETS backend
    COMPONENT Runtime
    RUNTIME_DEPENDENCIES
      PRE_EXCLUDE_REGEXES "libnv.*"  # TODO
      POST_EXCLUDE_REGEXES "^/usr/.*" "^/lib/.*"
  )
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/models
    COMPONENT Runtime
    DESTINATION .)
  # TODO
  add_custom_command(TARGET backend POST_BUILD
    COMMAND find ${opencv_INSTALL_DIR}/lib -maxdepth 1 -type f -exec patchelf --set-rpath '\$\$ORIGIN' {} \\\;
  )
endif()

