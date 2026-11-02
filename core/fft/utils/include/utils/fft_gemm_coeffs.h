/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "utils/fftensor_cache.h"

namespace AsdSip {

// 稠密 DFT 系数矩阵(GEMM 短维 pass 用)
FFTensor* BuildGemmDftTensor(int64_t fftLen, bool forward, int64_t stride = 0);

// 混合核 coeff:[W 垫宽 dim×stride | ex89 双向 tw 2080] 拼接为一张 GM 表
// (AIC GEMM 段与 AIV 融合变换段共用;内核按 dim*stride 定位 tw、按方向选 1040 段)
FFTensor* BuildHybridCoeffTensor(int64_t fftLen, bool forward, int64_t stride = 0);
// stride = 0 时取 RoundUp(2L, 8)(示例垫宽惯例);流水线用途显式传 2L(无垫列)

// N 维融合 pass 的 ex89 twiddle 表(500=20×25):[0,1040) forward / [1040,2080) inverse
// layout 与内核 NF_*_OFF 常量一致,移植自 example fft_coeffs.cpp
FFTensor* BuildFusedNTwiddleTensor();

} // namespace AsdSip
