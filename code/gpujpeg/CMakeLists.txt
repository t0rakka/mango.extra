cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(gpujpeg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Support matrix: Sources x Outputs (available = option ON + SDK found) ---
option(GPUJPEG_SOURCE_MANGO        "CPU mango::image decode source"                    ON)
option(GPUJPEG_SOURCE_MANGO_HYBRID "Mango Spectral Huffman → GPU compute (hybrid)"     ON)
option(GPUJPEG_SOURCE_NVJPEG       "NVIDIA nvJPEG decode source"                       ON)
option(GPUJPEG_SOURCE_ROC          "AMD rocJPEG decode source (VCN JPEG → HIP)"        ON)
if(APPLE)
    option(GPUJPEG_SOURCE_APPLE_VT "Apple VideoToolbox JPEG → IOSurface → VkImage" ON)
else()
    option(GPUJPEG_SOURCE_APPLE_VT "Apple VideoToolbox JPEG → IOSurface → VkImage" OFF)
endif()

option(GPUJPEG_OUTPUT_VULKAN "Publish intermediate image to Vulkan"     ON)
option(GPUJPEG_OUTPUT_OPENGL "Publish to OpenGL (stub)"                 OFF)
option(GPUJPEG_OUTPUT_D3D11  "Publish to D3D11 (stub)"                  OFF)
option(GPUJPEG_OUTPUT_D3D12  "Publish to D3D12 (stub)"                  OFF)

find_package(mango REQUIRED)

add_executable(gpujpeg
    src/main.cpp
    src/bench.hpp
    src/backends.hpp
    src/decoded_texture.hpp
    src/decoded_texture.cpp
    src/decoded_texture_hip.cpp
    src/nvjpeg_backend.hpp
    src/nvjpeg_backend.cpp
    src/mango_hybrid_backend.hpp
    src/mango_hybrid_backend.cpp
    src/mango_hybrid_compute.hpp
    src/mango_hybrid_compute.cpp
    src/mango_gpu_backend.hpp
    src/mango_gpu_backend.cpp
    src/roc_backend.hpp
    src/roc_backend.cpp
)

set(_gpujpeg_apple_sources "")
if(APPLE AND GPUJPEG_SOURCE_APPLE_VT)
    enable_language(OBJCXX)
    list(APPEND _gpujpeg_apple_sources
        src/apple_vt_backend.mm
        src/apple_imageio_backend.mm
        src/apple_vk_iosurface.mm)
    target_sources(gpujpeg PRIVATE ${_gpujpeg_apple_sources})
    target_compile_definitions(gpujpeg PRIVATE GPUJPEG_HAS_APPLE_VT=1)
    target_compile_definitions(gpujpeg PRIVATE GPUJPEG_HAS_SOURCE_APPLE_VT=1)
    target_link_libraries(gpujpeg PRIVATE
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreVideo"
        "-framework IOSurface"
        "-framework CoreImage"
        "-framework ImageIO"
        "-framework CoreGraphics"
        "-framework Foundation")
    message(STATUS "Source APPLE_VT: enabled (VideoToolbox + IOSurface import)")
elseif(GPUJPEG_SOURCE_APPLE_VT)
    message(STATUS "Source APPLE_VT: requested but not on APPLE — off")
else()
    message(STATUS "Source APPLE_VT: disabled")
endif()

target_include_directories(gpujpeg PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(gpujpeg PRIVATE
    mango::image
    mango::window
)

if (MSVC)
    target_compile_options(gpujpeg PRIVATE /W4 /Zc:__cplusplus /utf-8)
endif()

# --- Sources ---
if (GPUJPEG_SOURCE_MANGO)
    target_compile_definitions(gpujpeg PRIVATE GPUJPEG_HAS_SOURCE_MANGO=1)
    message(STATUS "Source MANGO: enabled")
else()
    message(STATUS "Source MANGO: disabled")
endif()

if (GPUJPEG_SOURCE_MANGO_HYBRID)
    if (NOT GPUJPEG_SOURCE_MANGO)
        message(FATAL_ERROR "GPUJPEG_SOURCE_MANGO_HYBRID requires GPUJPEG_SOURCE_MANGO=ON")
    endif()
    target_compile_definitions(gpujpeg PRIVATE GPUJPEG_HAS_SOURCE_MANGO_HYBRID=1)
    message(STATUS "Source MANGO_HYBRID: enabled (Huffman + VK compute)")
else()
    message(STATUS "Source MANGO_HYBRID: disabled")
endif()

set(_gpujpeg_has_nvjpeg FALSE)
if (GPUJPEG_SOURCE_NVJPEG)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND AND TARGET CUDA::nvjpeg)
        set(_gpujpeg_has_nvjpeg TRUE)
        target_compile_definitions(gpujpeg PRIVATE
            GPUJPEG_HAS_SOURCE_NVJPEG=1
            GPUJPEG_HAS_NVJPEG=1)
        target_link_libraries(gpujpeg PRIVATE CUDA::cudart CUDA::nvjpeg CUDA::nppc CUDA::nppidei)
        message(STATUS "Source NVJPEG: enabled (${CUDAToolkit_VERSION})")
    else()
        message(STATUS "Source NVJPEG: requested but CUDA/nvJPEG not found — off")
    endif()
else()
    message(STATUS "Source NVJPEG: disabled")
endif()

set(_gpujpeg_has_roc FALSE)
if (GPUJPEG_SOURCE_ROC)
    set(_roc_roots "")
    if (DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        list(APPEND _roc_roots "$ENV{ROCM_PATH}")
    endif()
    list(APPEND _roc_roots /opt/rocm)

    find_path(ROCJPEG_INCLUDE_DIR rocjpeg/rocjpeg.h
        PATHS ${_roc_roots}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)
    find_library(ROCJPEG_LIBRARY NAMES rocjpeg
        PATHS ${_roc_roots}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)
    find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
        PATHS ${_roc_roots}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)
    find_library(HIP_RUNTIME_LIBRARY NAMES amdhip64
        PATHS ${_roc_roots}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)
    find_library(HIPRTC_LIBRARY NAMES hiprtc
        PATHS ${_roc_roots}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)

    if (ROCJPEG_INCLUDE_DIR AND ROCJPEG_LIBRARY AND HIP_INCLUDE_DIR
            AND HIP_RUNTIME_LIBRARY AND HIPRTC_LIBRARY)
        set(_gpujpeg_has_roc TRUE)
        # GCC does not set a HIP platform (only hipcc does). Required before hip_runtime.h,
        # including in TUs that also include CUDA headers.
        target_compile_definitions(gpujpeg PRIVATE
            GPUJPEG_HAS_SOURCE_ROC=1
            GPUJPEG_HAS_ROC=1
            __HIP_PLATFORM_AMD__)
        target_include_directories(gpujpeg PRIVATE
            ${ROCJPEG_INCLUDE_DIR}
            ${HIP_INCLUDE_DIR})
        target_link_libraries(gpujpeg PRIVATE
            ${ROCJPEG_LIBRARY}
            ${HIP_RUNTIME_LIBRARY}
            ${HIPRTC_LIBRARY})
        get_filename_component(_roc_libdir "${HIP_RUNTIME_LIBRARY}" DIRECTORY)
        set_property(TARGET gpujpeg APPEND PROPERTY BUILD_RPATH "${_roc_libdir}")
        set_property(TARGET gpujpeg APPEND PROPERTY INSTALL_RPATH "${_roc_libdir}")
        message(STATUS "Source ROC: enabled (${ROCJPEG_LIBRARY})")
    else()
        message(STATUS "Source ROC: requested but rocJPEG/HIP SDK not found — off")
        message(STATUS "  looked for rocjpeg.h, librocjpeg, hip_runtime.h, libamdhip64, libhiprtc")
        message(STATUS "  under ROCM_PATH and /opt/rocm (Arch: pacman -S rocjpeg)")
    endif()
else()
    message(STATUS "Source ROC: disabled")
endif()

# --- Outputs ---
set(_gpujpeg_has_vulkan FALSE)
if (GPUJPEG_OUTPUT_VULKAN)
    set(_gpujpeg_has_vulkan TRUE)
    target_compile_definitions(gpujpeg PRIVATE GPUJPEG_HAS_OUTPUT_VULKAN=1)
    target_link_libraries(gpujpeg PRIVATE mango::vulkan)
    message(STATUS "Output VULKAN: enabled")
else()
    message(FATAL_ERROR "Workbench currently requires GPUJPEG_OUTPUT_VULKAN=ON")
endif()

if (GPUJPEG_OUTPUT_OPENGL)
    message(STATUS "Output OPENGL: stub (not implemented)")
else()
    message(STATUS "Output OPENGL: disabled")
endif()

if (WIN32 AND GPUJPEG_OUTPUT_D3D11)
    message(STATUS "Output D3D11: stub (not implemented)")
else()
    message(STATUS "Output D3D11: disabled")
endif()

if (WIN32 AND GPUJPEG_OUTPUT_D3D12)
    message(STATUS "Output D3D12: stub (not implemented)")
else()
    message(STATUS "Output D3D12: disabled")
endif()

# Copy CUDA runtime DLLs next to the exe on Windows.
if (WIN32 AND _gpujpeg_has_nvjpeg)
    set(_cuda_bin "${CUDAToolkit_BIN_DIR}")
    if (EXISTS "${CUDAToolkit_ROOT_DIR}/bin/x64/cudart64_13.dll")
        set(_cuda_bin "${CUDAToolkit_ROOT_DIR}/bin/x64")
    elseif (EXISTS "${CUDAToolkit_TARGET_DIR}/bin/x64/cudart64_13.dll")
        set(_cuda_bin "${CUDAToolkit_TARGET_DIR}/bin/x64")
    endif()

    add_custom_command(TARGET gpujpeg POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_cuda_bin}/cudart64_13.dll"
            "${_cuda_bin}/nvjpeg64_13.dll"
            "${_cuda_bin}/nppc64_13.dll"
            "${_cuda_bin}/nppidei64_13.dll"
            $<TARGET_FILE_DIR:gpujpeg>
        COMMENT "Copy CUDA/nvJPEG/NPP runtime DLLs beside gpujpeg"
    )
endif()
