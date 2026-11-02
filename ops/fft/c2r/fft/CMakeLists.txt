# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(fft_c2r_src
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2r_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft/fft_c2r_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft/tiling/fft_c2r_tiling.cpp
)

set(CATLASS_INCLUDES
    ${OPS_THIRD_PARTY_DIR}/catlass/include
    $ENV{ASCEND_HOME_PATH}/include
    ${PROJECT_SOURCE_DIR}/ops/include/template
)

add_operation(FftC2ROperation "${fft_c2r_src}")


add_kernel(fft_c2r_1 ascend910b mix
    fft/kernel/fft_c2r_1.cce
    FftC2R1C64Kernel)

add_kernel(fft_c2r_2 ascend910b mix
    fft/kernel/fft_c2r_2.cce
    FftC2R2C64Kernel)

add_kernel(fft_c2r_3 ascend910b mix
    fft/kernel/fft_c2r_3.cce
    FftC2R3C64Kernel)

add_kernel(fft_c2r_1_odd ascend910b mix
    fft/kernel/fft_c2r_1_odd.cce
    FftC2ROdd1C64Kernel)

add_kernel(fft_c2r_2_odd ascend910b mix
    fft/kernel/fft_c2r_2_odd.cce
    FftC2ROdd2C64Kernel)

add_kernel(fft_c2r_3_odd ascend910b mix
    fft/kernel/fft_c2r_3_odd.cce
    FftC2ROdd3C64Kernel)

add_kernel(fft_c2r_1_even ascend910b mix
    fft/kernel/fft_c2r_1_even.cce
    FftC2REven1C64Kernel)

add_kernel(fft_c2r_2_even ascend910b mix
    fft/kernel/fft_c2r_2_even.cce
    FftC2REven2C64Kernel)

add_kernel(fft_c2r_3_even ascend910b mix
    fft/kernel/fft_c2r_3_even.cce
    FftC2REven3C64Kernel)

# Ascend 950 (arch35) kernels using Catlass
add_kernel(fft_c2r_multi_core ascend950 vector
    fft/kernel/arch35/fft_c2r_multi_core.cpp
    FftC2RC64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})
