# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(fft_c2c_arch35_src
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35/fft_c2c_arch35_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35/tiling/fft_c2c_arch35_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35/tiling/fft_c2c_arch35_mix_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35/tiling/fft_c2c_arch35_gemm_tiling.cpp
)

# catlass overlay 以 .h 存放(命名规范);构建期生成同内容 .hpp 副本,保持对 3rdparty 克隆的同路径 include 遮蔽
# 仅 tile_mmad 一件:3510 的 PipeBarrier<PIPE_M> 编译器兼容守卫(GEMM 核的 BlockMmadTla 链必需)
set(CATLASS_EXT_DIR ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35/catlass_ext)
set(CATLASS_EXT_GEN ${CMAKE_CURRENT_BINARY_DIR}/catlass_ext_gen)
configure_file(${CATLASS_EXT_DIR}/catlass/gemm/tile/tile_mmad.h ${CATLASS_EXT_GEN}/catlass/gemm/tile/tile_mmad.hpp COPYONLY)

set(CATLASS_INCLUDES
    ${CATLASS_EXT_GEN}
    ${OPS_THIRD_PARTY_DIR}/catlass/include
    $ENV{ASCEND_HOME_PATH}/include
    ${PROJECT_SOURCE_DIR}/ops/include/template
)

add_operation(FftC2CArch35Operation "${fft_c2c_arch35_src}")

add_kernel(fft_c2c_arch35_multi_core ascend950 vector
    fft_c2c_arch35/kernel/arch35/fft_c2c_multi_core.cpp
    FftC2CArch35C64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})

add_kernel(fft_c2c_arch35_mix_multi_core ascend950 vector
    fft_c2c_arch35/kernel/arch35/fft_c2c_mix_multi_core.cpp
    FftC2CArch35MixC64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})

add_kernel(fft_c2c_arch35_gemm_multi_core ascend950 cube
    fft_c2c_arch35/kernel/arch35/fft_c2c_gemm_multi_core.cpp
    FftC2CArch35GemmC64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})

add_kernel(fft_c2c_arch35_trans_multi_core ascend950 vector
    fft_c2c_arch35/kernel/arch35/fft_c2c_arch35_trans_multi_core.cpp
    FftC2CArch35TransC64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})

add_kernel(fft_c2c_arch35_nfused_multi_core ascend950 vector
    fft_c2c_arch35/kernel/arch35/fft_c2c_arch35_nfused_multi_core.cpp
    FftC2CArch35NFusedC64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})

add_kernel(fft_c2c_arch35_hybrid_multi_core ascend950 mix
    fft_c2c_arch35/kernel/arch35/fft_c2c_arch35_hybrid_multi_core.asc
    FftC2CArch35HybridC64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})
