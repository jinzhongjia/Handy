cmake_minimum_required(VERSION 3.21)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
endif()
cmake_policy(SET CMP0091 NEW)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
project(handy_xasr LANGUAGES C CXX)
include(FetchContent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)

# Never compile for the build machine's ISA. ORT provides its own CPU dispatch.
if(MSVC)
  add_compile_options(/EHsc /utf-8)
elseif(HANDY_XASR_TARGET_ARCH STREQUAL "x86_64")
  add_compile_options(-march=x86-64 -mtune=generic)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_options(-march=armv8-a)
endif()

if(HANDY_XASR_DEPENDENCY_DIR)
  # Upstream recipes look in CMAKE_BINARY_DIR and verify their pinned SHA256.
  file(GLOB cached_archives "${HANDY_XASR_DEPENDENCY_DIR}/*.tar.gz"
    "${HANDY_XASR_DEPENDENCY_DIR}/*.tar.bz2" "${HANDY_XASR_DEPENDENCY_DIR}/*.zip")
  foreach(archive IN LISTS cached_archives)
    file(COPY "${archive}" DESTINATION "${CMAKE_BINARY_DIR}")
  endforeach()
endif()
# Cargo's build driver downloads/hash-checks the archive and extracts regular
# files only; CMake's generic extraction would retain unsafe upstream symlinks.
set(sherpa_SOURCE_DIR "${HANDY_XASR_SOURCE_DIR}")
if(NOT EXISTS "${sherpa_SOURCE_DIR}/sherpa-onnx/csrc/online-recognizer.h")
  message(FATAL_ERROR "HANDY_XASR_SOURCE_DIR must point at the pinned sherpa source root")
endif()

# Load exactly one SHARED ORT while keeping all sherpa components STATIC.
# Variables below tell sherpa's own onnxruntime.cmake not to create another target.
set(SHERPA_ONNX_USE_STATIC_CRT OFF CACHE BOOL "" FORCE)
if(HANDY_ORT_ROOT)
  if(HANDY_ORT_INCLUDE_DIR)
    set(ort_header_roots "${HANDY_ORT_INCLUDE_DIR}")
  else()
    set(ort_header_roots "${HANDY_ORT_ROOT}/include" "${HANDY_ORT_ROOT}/../include")
  endif()
  find_path(ort_include onnxruntime_cxx_api.h PATHS ${ort_header_roots}
    PATH_SUFFIXES "" onnxruntime onnxruntime/core/session NO_DEFAULT_PATH REQUIRED)
  if(WIN32)
    find_file(ort_runtime onnxruntime.dll PATHS "${HANDY_ORT_ROOT}" "${HANDY_ORT_ROOT}/lib"
      "${HANDY_ORT_ROOT}/bin" "${HANDY_ORT_ROOT}/../bin" NO_DEFAULT_PATH REQUIRED)
    find_library(ort_import onnxruntime PATHS "${HANDY_ORT_ROOT}" "${HANDY_ORT_ROOT}/lib"
      NO_DEFAULT_PATH REQUIRED)
  elseif(APPLE)
    find_file(ort_runtime libonnxruntime.dylib PATHS "${HANDY_ORT_ROOT}" "${HANDY_ORT_ROOT}/lib"
      NO_DEFAULT_PATH REQUIRED)
  else()
    find_file(ort_runtime NAMES libonnxruntime.so.1 libonnxruntime.so PATHS
      "${HANDY_ORT_ROOT}" "${HANDY_ORT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  endif()
  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ort_runtime}"
    INTERFACE_INCLUDE_DIRECTORIES "${ort_include}")
  if(WIN32)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_IMPLIB "${ort_import}")
  endif()
  set(ort_notice_roots "${HANDY_ORT_ROOT}" "${HANDY_ORT_ROOT}/..")
else()
  # These recipes are part of the immutable sherpa archive and each pins the
  # runtime archive SHA256. They fetch ORT only, never a stock sherpa/TTS build.
  set(BUILD_SHARED_LIBS ON)
  if(APPLE)
    if(HANDY_XASR_TARGET_ARCH STREQUAL "aarch64")
      set(ort_platform osx-arm64)
    else()
      set(ort_platform osx-x86_64)
    endif()
  elseif(WIN32)
    if(HANDY_XASR_TARGET_ARCH STREQUAL "aarch64")
      set(ort_platform win-arm64)
      set(CMAKE_VS_PLATFORM_NAME ARM64)
    else()
      set(ort_platform win-x64)
      set(CMAKE_VS_PLATFORM_NAME x64)
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ort_platform "linux-${HANDY_XASR_TARGET_ARCH}")
  else()
    message(FATAL_ERROR "Unsupported X-ASR platform ${CMAKE_SYSTEM_NAME}")
  endif()
  include("${sherpa_SOURCE_DIR}/cmake/onnxruntime-${ort_platform}.cmake")
  set(ort_include "${onnxruntime_SOURCE_DIR}/include")
  if(WIN32)
    set(ort_runtime "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll")
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ort_runtime}")
  else()
    get_target_property(ort_runtime onnxruntime IMPORTED_LOCATION)
  endif()
  set(ort_notice_roots "${onnxruntime_SOURCE_DIR}")
endif()
if(HANDY_ORT_LICENSE_DIR)
  list(PREPEND ort_notice_roots "${HANDY_ORT_LICENSE_DIR}")
endif()
# Compile-time guard; the runtime entry point additionally requests API 24 before
# any native recognizer is constructed or any Rust ort API is used.
file(STRINGS "${ort_include}/onnxruntime_c_api.h" ort_api_line REGEX "^#define ORT_API_VERSION [0-9]+")
string(REGEX MATCH "[0-9]+" ort_api_version "${ort_api_line}")
if(NOT ort_api_version OR ort_api_version LESS 24)
  message(FATAL_ERROR "X-ASR needs ONNX Runtime API >=24 headers and a compatible shared library")
endif()
set(location_onnxruntime_header_dir "${ort_include}")
set(location_onnxruntime_lib "${ort_runtime}")
set(onnxruntime_lib_files "${ort_runtime}")
set(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
foreach(option ENABLE_TTS ENABLE_SPEAKER_DIARIZATION ENABLE_BINARY ENABLE_C_API
    ENABLE_PYTHON ENABLE_TESTS ENABLE_CHECK ENABLE_JNI ENABLE_PORTAUDIO ENABLE_WEBSOCKET
    ENABLE_GPU ENABLE_DIRECTML ENABLE_RKNN ENABLE_AXERA ENABLE_AXCL ENABLE_ASCEND_NPU
    ENABLE_QNN ENABLE_SPACEMIT BUILD_C_API_EXAMPLES LINK_LIBSTDCPP_STATICALLY)
  set(SHERPA_ONNX_${option} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory("${sherpa_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/sherpa" EXCLUDE_FROM_ALL)
# The bridge is CPU-only, including on macOS. Supplied CPU SDKs need not
# provide CoreML's optional factory header or exported provider entry point.
target_compile_definitions(sherpa-onnx-core PRIVATE SHERPA_ONNX_DISABLE_COREML=1)

add_library(handy_xasr SHARED handy_xasr.cpp)
target_include_directories(handy_xasr PRIVATE "${sherpa_SOURCE_DIR}" "${ort_include}")
target_link_libraries(handy_xasr PRIVATE sherpa-onnx-core onnxruntime)
if(APPLE)
  set_target_properties(handy_xasr PROPERTIES MACOSX_RPATH ON INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(handy_xasr PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")
endif()

# Preserve notices for all statically linked dependencies as well as shared ORT.
# Do not install sherpa's broad install target: only the wrapper, ORT and notices.
file(MAKE_DIRECTORY "${HANDY_XASR_RUNTIME_DIR}/licenses")
function(copy_notices name)
  set(found FALSE)
  foreach(root IN LISTS ARGN)
    file(GLOB notices "${root}/LICENSE*" "${root}/COPYING*" "${root}/NOTICE*"
      "${root}/Copyright*" "${root}/copyright*" "${root}/ThirdPartyNotices*")
    foreach(notice IN LISTS notices)
      file(COPY "${notice}" DESTINATION "${HANDY_XASR_RUNTIME_DIR}/licenses/${name}")
      set(found TRUE)
    endforeach()
  endforeach()
  if(NOT found)
    message(FATAL_ERROR "Missing redistribution notices for ${name}; for supplied ORT set ORT_LICENSE_DIR")
  endif()
endfunction()
copy_notices(sherpa-onnx "${sherpa_SOURCE_DIR}")
copy_notices(onnxruntime ${ort_notice_roots})
foreach(dependency kaldi_native_fbank kaldi_decoder kaldifst openfst kissfft simple-sentencepiece json eigen)
  FetchContent_GetProperties(${dependency} SOURCE_DIR dependency_source)
  copy_notices("${dependency}" "${dependency_source}")
endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_program(HANDY_PATCHELF patchelf REQUIRED)
endif()
configure_file(StageRuntime.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/StageRuntime.cmake" @ONLY)
add_custom_target(handy_xasr_stage
  COMMAND "${CMAKE_COMMAND}" "-DWRAPPER=$<TARGET_FILE:handy_xasr>"
    "-DIMPORT_LIBRARY=$<TARGET_LINKER_FILE:handy_xasr>"
    -P "${CMAKE_CURRENT_BINARY_DIR}/StageRuntime.cmake"
  DEPENDS handy_xasr
  VERBATIM)
