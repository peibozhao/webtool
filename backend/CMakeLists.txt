
cmake_minimum_required(VERSION 3.10)
project(webtool)
set(CMAKE_CXX_STANDARD 20)
add_definitions(-DSPDLOG_USE_STD_FORMAT)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/spdlog/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cpp-httplib/)
file(GLOB_RECURSE BACKEND_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(backend ${BACKEND_SOURCES})

