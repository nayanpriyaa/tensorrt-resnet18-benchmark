cmake_minimum_required(VERSION 3.18)
project(trt_resnet18 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------------------------------------
# CUDA runtime. Only the host-side runtime API is used (no .cu files), so the project compiles with
# the system C++ compiler (GCC/Clang/MSVC) and links cudart; nvcc only helps CMake locate the toolkit.
#   Linux / WSL2 : /usr/local/cuda            Windows : found via the CUDA_PATH env var the installer sets
# Override with -DCUDAToolkit_ROOT=/path/to/cuda.
# ------------------------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------------------------
# TensorRT
#   Linux deb (apt)   : headers/libs in /usr/include|lib/<arch>-linux-gnu, found automatically
#   Linux tar         : -DTENSORRT_ROOT=/opt/TensorRT-10.x.y          (or export TENSORRT_ROOT)
#   Windows zip       : -DTENSORRT_ROOT=C:/TensorRT-10.x.y.z          headers in include\, import libs in
#                       lib\ (nvinfer_10.lib / nvinfer_11.lib), DLLs in bin\ or lib\ depending on release:
#                       that folder must be on PATH at run time.
# ------------------------------------------------------------------------------------------------
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT install prefix (tar/zip installs)")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES include include/${CMAKE_LIBRARY_ARCHITECTURE})

find_library(TENSORRT_NVINFER_LIBRARY
  NAMES nvinfer nvinfer_11 nvinfer_10
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/${CMAKE_LIBRARY_ARCHITECTURE})

find_library(TENSORRT_NVONNXPARSER_LIBRARY
  NAMES nvonnxparser nvonnxparser_11 nvonnxparser_10
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/${CMAKE_LIBRARY_ARCHITECTURE})

# Version from the header. TRT >= 10.x defines NV_TENSORRT_MAJOR via TRT_MAJOR_ENTERPRISE.
if(TENSORRT_INCLUDE_DIR AND EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_version_lines
       REGEX "#define (NV_TENSORRT|TRT)_(MAJOR|MINOR|PATCH)(_ENTERPRISE)? +[0-9]+")
  foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "#define (NV_TENSORRT_${_part}|TRT_${_part}_ENTERPRISE) +([0-9]+)" _match "${_trt_version_lines}")
    set(TensorRT_VERSION_${_part} "${CMAKE_MATCH_2}")
  endforeach()
  set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TENSORRT_INCLUDE_DIR TENSORRT_NVINFER_LIBRARY TENSORRT_NVONNXPARSER_LIBRARY
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_VERSION_MAJOR VERSION_LESS 10)
  message(FATAL_ERROR "TensorRT ${TensorRT_VERSION} found; this project uses the TRT 10+ API "
                      "(named I/O tensors, enqueueV3).")
endif()
if(TensorRT_VERSION_MAJOR VERSION_GREATER_EQUAL 11)
  message(STATUS "TensorRT ${TensorRT_VERSION}: strongly typed only -> FP16 runs need resnet18_fp16.onnx "
                 "(python export_to_onnx.py --fp16-onnx)")
endif()

# Imported targets: include dirs of IMPORTED targets are treated as SYSTEM, so -Wall/-Wextra below only
# applies to this project's code, not to the TensorRT/CUDA headers.
add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvinfer PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES CUDA::cudart)          # NvInfer.h includes cuda_runtime_api.h

add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvonnxparser PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_NVONNXPARSER_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)

set(TRTP_WARNINGS
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /utf-8>)
# Defensive: if anything in the include chain ever pulls in windows.h, its min/max macros break std::min.
set(TRTP_DEFINES $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX WIN32_LEAN_AND_MEAN>)

# ------------------------------------------------------------------------------------------------
# Library: engine build + inference wrapper
# ------------------------------------------------------------------------------------------------
add_library(trt_pipeline STATIC TrtPipeline.cpp)
target_include_directories(trt_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trt_pipeline
  PUBLIC  TensorRT::nvinfer CUDA::cudart
  PRIVATE TensorRT::nvonnxparser)                 # parser is a build-time dependency of TrtPipeline.cpp only
target_compile_options(trt_pipeline PRIVATE ${TRTP_WARNINGS})
target_compile_definitions(trt_pipeline PUBLIC ${TRTP_DEFINES})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  target_link_libraries(trt_pipeline PUBLIC stdc++fs)   # std::filesystem on GCC 8
endif()

# ------------------------------------------------------------------------------------------------
# Benchmark executable
# ------------------------------------------------------------------------------------------------
add_executable(trt_resnet18 main.cpp)
target_link_libraries(trt_resnet18 PRIVATE trt_pipeline)
target_compile_options(trt_resnet18 PRIVATE ${TRTP_WARNINGS})

message(STATUS "CUDA      : ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_DIR})")
message(STATUS "TensorRT  : ${TensorRT_VERSION} (${TENSORRT_NVINFER_LIBRARY})")
