# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(sgetri_batched_src
    ${CMAKE_CURRENT_LIST_DIR}/sgetri_batched_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sgetri_batched/sgetri_batched_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sgetri_batched/tiling/sgetri_batched_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sgetri_p2_batched/sgetri_p2_batched_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sgetri_p2_batched/tiling/sgetri_p2_batched_tiling.cpp
)

add_operation(SgetriBatchedOperation "${sgetri_batched_src}")

add_kernel(sgetri_batched ascend910b mix
    sgetri_batched/kernel/sgetri_batched.cpp
    SgetriBatchedF32Kernel)

# P2 每核独立路径 (实数 n<=256): 纯 AIV kernel, 零跨核同步
add_kernel(sgetri_p2_batched ascend910b vector
    sgetri_p2_batched/kernel/sgetri_p2_batched.cpp
    SgetriP2BatchedF32Kernel)
