# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(real_stockham_dir ${CMAKE_CURRENT_LIST_DIR})
add_operation(FftRealStockhamOperation "${real_stockham_dir}/fft_stockham_operation.cpp")
# 展开为字面类名(扫描器不解析 CMake 变量;见交接文档十四章)
add_kernel(fft_real_stockham_c2r ascend950 vector
    fft_stockham/kernel/arch35/fft_real_stockham_c2r.cpp FftRealStockhamC2RKernel
    CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include
    DEPENDS ${real_stockham_dir}/fft_stockham/kernel/arch35/vendor/fft_stockham_mix.h
            ${PROJECT_SOURCE_DIR}/ops/include/params/fft_real_stockham_tiling.h)
add_kernel(fft_real_stockham_r2c ascend950 vector
    fft_stockham/kernel/arch35/fft_real_stockham_r2c.cpp FftRealStockhamR2CKernel
    CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include
    DEPENDS ${real_stockham_dir}/fft_stockham/kernel/arch35/vendor/fft_stockham_mix.h
            ${PROJECT_SOURCE_DIR}/ops/include/params/fft_real_stockham_tiling.h)
