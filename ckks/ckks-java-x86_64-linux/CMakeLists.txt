cmake_minimum_required(VERSION 3.17)
project(bigdl-he-poc)
set(CMAKE_CXX_STANDARD 17) # seal requires C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# option(ENABLE_INTEL_HEXL "Enable Intel HEXL library" OFF)
include(cmake/seal.cmake)

find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)

include_directories(../src
      ${JNI_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${CMAKE_CURRENT_BINARY_DIR}")

add_library(vflhe_kms ../src/main/c/kms.cpp ../src/main/c/common.h)
target_link_libraries(vflhe_kms libseal)

add_library(vflhe_data_encryptor ../src/main/c/data_encryptor.cpp ../src/main/c/common.h)
target_link_libraries(vflhe_data_encryptor libseal)

add_library(vflhe_ckks_common ../src/main/c/ckks_common.cpp ../src/main/c/common.h)
target_link_libraries(vflhe_ckks_common libseal)

add_library(vflhe_jni SHARED ../src/main/c/com_intel_analytics_bigdl_ckks_CKKS.cpp ../src/main/c/kms.h  ../src/main/c/com_intel_analytics_bigdl_ckks_CKKS.h ../src/main/c/common.h ../src/main/c/ckks_common.h  ../src/main/c/data_encryptor.h)
target_link_libraries(vflhe_jni libseal vflhe_kms vflhe_data_encryptor vflhe_ckks_common ${JNI_LIBRARIES} pthread -static-libstdc++)
