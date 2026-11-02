# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(fft_r2c_src
    ${CMAKE_CURRENT_LIST_DIR}/fft_r2c_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft/fft_r2c_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft/tiling/fft_r2c_tiling.cpp
)

add_operation(FftR2COperation "${fft_r2c_src}")


add_kernel(fft_r2c_1 ascend910b mix
    fft/kernel/fft_r2c_1.cce
    FftR2C1C64Kernel)

add_kernel(fft_r2c_2 ascend910b mix
    fft/kernel/fft_r2c_2.cce
    FftR2C2C64Kernel)

add_kernel(fft_r2c_3 ascend910b mix
    fft/kernel/fft_r2c_3.cce
    FftR2C3C64Kernel)

add_kernel(fft_r2c_1_odd ascend910b mix
    fft/kernel/fft_r2c_1_odd.cce
    FftR2COdd1C64Kernel)

add_kernel(fft_r2c_2_odd ascend910b mix
    fft/kernel/fft_r2c_2_odd.cce
    FftR2COdd2C64Kernel)

add_kernel(fft_r2c_3_odd ascend910b mix
    fft/kernel/fft_r2c_3_odd.cce
    FftR2COdd3C64Kernel)

add_kernel(fft_r2c_1_even ascend910b mix
    fft/kernel/fft_r2c_1_even.cce
    FftR2CEven1C64Kernel)

add_kernel(fft_r2c_2_even ascend910b mix
    fft/kernel/fft_r2c_2_even.cce
    FftR2CEven2C64Kernel)

add_kernel(fft_r2c_3_even ascend910b mix
    fft/kernel/fft_r2c_3_even.cce
    FftR2CEven3C64Kernel)

add_kernel(fft_r2c_multi_core ascend950 vector
    fft/kernel/arch35/fft_r2c_multi_core.cpp
    FftR2CC64Kernel)
