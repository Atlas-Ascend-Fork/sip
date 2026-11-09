# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 设置链接库地址
set(LINKED_LIBRARIES blas base fft filter interpolation utils asdsip_core dl c_sec mmpa)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 包括PyTorch头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/core/include)
include_directories(${PROJECT_SOURCE_DIR}/core/utils/include)

# st/ 与 fuzz/ 测试目录按规划建设中，尚不存在（issue #367）：按目录存在性守卫，
# 避免 add_subdirectory 指向不存在目录导致 TEST_TYPE=ST/FT 配置阶段直接失败；
# 目录缺失时给出明确提示并跳过，后续目录就位后无需改动本文件即可接入
if("$ENV{TEST_TYPE}" STREQUAL "UT")
    add_subdirectory(ut)
endif()
if("$ENV{TEST_TYPE}" STREQUAL "ST")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/st/CMakeLists.txt")
        add_subdirectory(st)
    else()
        message(WARNING "TEST_TYPE=ST requested, but tests/st does not exist; ST tests are skipped. Currently only TEST_TYPE=UT is available.")
    endif()
endif()
if("$ENV{TEST_TYPE}" STREQUAL "FT")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fuzz/CMakeLists.txt")
        add_subdirectory(fuzz)
    else()
        message(WARNING "TEST_TYPE=FT requested, but tests/fuzz does not exist; fuzz tests are skipped. Currently only TEST_TYPE=UT is available.")
    endif()
endif()
