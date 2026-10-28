# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(ifftshift_src
    ${CMAKE_CURRENT_LIST_DIR}/ifftshift_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ifftshift/ifftshift_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ifftshift/tiling/ifftshift_tiling.cpp
)

add_operation(IfftShiftOperation "${ifftshift_src}")

# Single unified kernel: a runtime-col 3-stage fast pipeline (whole-row MTE2
# read -> vector repack -> whole-row MTE3 write) with a piecewise fallback for
# very wide columns; dispatch happens inside the kernel (see kernel/ifftshift.cpp).
add_kernel(IfftShift ascend950 vector
    ifftshift/kernel/ifftshift.cpp
    IfftShiftC64Kernel)
