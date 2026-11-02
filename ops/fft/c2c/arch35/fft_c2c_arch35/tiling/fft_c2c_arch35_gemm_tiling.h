/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDSIP_FFT_C2C_ARCH35_GEMM_TILING_H
#define ASDSIP_FFT_C2C_ARCH35_GEMM_TILING_H

#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include "params/fft_c2c_arch35.h"
#include "utils/aspb_status.h"

namespace AsdSip {
using namespace Mki;

// GEMM 短维 pass 的设备侧 tiling(POD,核从 GM 直读;host/device 共享)
struct FftGemmTilingData {
    int64_t rows; // 该维变换行数(batch × 另一维)
    int32_t kDim; // 2L(实虚交错后的 K)
    int32_t nDim; // 2L(无垫列:GEMM C 即标准 [b][x][ỹ] 复数布局)
    int32_t isInverse;
    uint32_t coreNum; // 启动的 AIC 数
};

// 收尾转置 pass 的设备侧 tiling:[batch, fftN, fftM] → [batch, fftM, fftN]
struct FftTransTilingData {
    int64_t batch;
    int32_t fftN;
    int32_t fftM;
};

// N 维融合 pass 的设备侧 tiling(与内核 NFusedTilingData 逐字段一致)
struct FftNFusedTilingData {
    int64_t batch;
    int32_t fftM;       // 每矩阵变换条数(瓦片行数,1 至 52)
    int32_t direction;  // 1=forward
    int32_t twFloatOff; // 方向已选的 ex89 twiddle 表偏移(浮点)
    int32_t coreNum;
};

// 混合核设备侧 tiling:__mix__(1,2),AIC GEMM→owner AIV UB 瓦片,AIV 排空回 GM
// 。设计、旗标协议与 UB 行距 32B 对齐规则
struct FftHybridTilingData {
    int64_t batch; // 矩阵数(每矩阵 = 一个 [fftN, fftM] 复数块)
    int32_t fftN;  // 长维:瓦片行数
    int32_t fftM;  // 短维:每行复数点数(GEMM 的 L)
    int32_t nDim;  // GEMM C 行宽 = RoundUp8(2*fftM)(Fixpipe UB 行距 32B 对齐)
    int32_t isInverse;
    uint32_t coreNum;  // AIC 数(mix 核 blockDim)
    int32_t debugMode; // 诊断模式:0=关闭(默认);1=AIC 旁路;2=AIV 旁路;3=双侧旁路
};

AspbStatus FftC2CArch35GemmTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo);
AspbStatus FftC2CArch35TransTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo);
AspbStatus FftC2CArch35NFusedTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo);
AspbStatus FftC2CArch35HybridTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo);

} // namespace AsdSip

#endif
