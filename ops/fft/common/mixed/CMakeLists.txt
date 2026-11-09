# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_operation(FftMixedCommonOperation operation.cpp)
add_kernel(fft_mixed_common ascend950 vector kernel/mixed.asc FftMixedCommonKernel
    CACHE_POLICY compiler-default
    INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/kernel/row_fft.h
        ${CMAKE_CURRENT_LIST_DIR}/kernel/radix3.h
        ${CMAKE_CURRENT_LIST_DIR}/kernel/radix4.h
        ${CMAKE_CURRENT_LIST_DIR}/kernel/radix57.h
        ${CMAKE_CURRENT_LIST_DIR}/kernel/odd_symmetric.h
        ${PROJECT_SOURCE_DIR}/ops/include/params/fft_mixed_common.h)
