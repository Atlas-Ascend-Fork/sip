# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ASC REQUIRED)

project(fft_2d_c2r_4096x4096 LANGUAGES ASC CXX)

set(FFT_SOURCES
    fft_c2r_2d.cpp
    fft_plan.cpp
    fft_tiling.cpp
    fft_kernel.cpp
)

set_source_files_properties(${FFT_SOURCES} PROPERTIES LANGUAGE ASC)

add_executable(fft_2d_c2r_4096x4096 ${FFT_SOURCES})

set_target_properties(fft_2d_c2r_4096x4096 PROPERTIES LINKER_LANGUAGE ASC)

target_include_directories(fft_2d_c2r_4096x4096 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
)

target_compile_options(fft_2d_c2r_4096x4096 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    $<$<COMPILE_LANGUAGE:ASC>:--cce-auto-sync=off>
    $<$<COMPILE_LANGUAGE:ASC>:-DENABLE_CV_COMM_VIA_SSBUF=true>
)

target_compile_definitions(fft_2d_c2r_4096x4096 PRIVATE
    CATLASS_ARCH=3510
    ENABLE_CV_COMM_VIA_SSBUF=true
)

link_directories(fft_2d_c2r_4096x4096 PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)

target_link_libraries(fft_2d_c2r_4096x4096 PRIVATE
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
    tiling_api nnopbase platform
    dl
    m
)

if(EXISTS $ENV{ASCEND_HOME_PATH}/lib64/libascendalog.so)
    target_link_libraries(fft_2d_c2r_4096x4096 PRIVATE ascendalog)
endif()
if(EXISTS $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
    target_link_libraries(fft_2d_c2r_4096x4096 PRIVATE unified_dlog)
endif()

# HAL 库路径按主机架构推导（对齐主仓 build.sh 的 ${HOSTTYPE}-linux 惯例），
# 避免 aarch64 环境链接到不存在的 x86_64-linux 路径
set(ASDSIP_HOST_ARCH_LINUX "x86_64-linux")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(ASDSIP_HOST_ARCH_LINUX "aarch64-linux")
endif()
target_link_libraries(fft_2d_c2r_4096x4096 PRIVATE $ENV{ASCEND_HOME_PATH}/${ASDSIP_HOST_ARCH_LINUX}/devlib/libascend_hal.so)
