/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "getri_custom.hpp"

static constexpr uint32_t OFFSET_TWO = 2;
static constexpr uint32_t OFFSET_THREE = 3;
static constexpr uint32_t OFFSET_FOUR = 4;
static constexpr uint32_t OFFSET_FIVE = 5;

static constexpr int64_t EYE_MATRIX_NUM = 3;
static constexpr int64_t BASE_BLOCK_ELENUM = 16;
static constexpr int64_t COL_ALIGNED_ELENUM = 128;
static constexpr int64_t TILE_ELENUM = 512;
static constexpr int64_t COMPLEX_ELENUM = 2;

extern "C" __global__ __aicore__ void sgetri_batched(GM_ADDR sync, GM_ADDR A_org, GM_ADDR W, GM_ADDR gather1_gm,
                                                     GM_ADDR gather2_gm, GM_ADDR gather3_gm, GM_ADDR eye_gm,
                                                     GM_ADDR A_work, GM_ADDR work_gm, GM_ADDR A_inv, GM_ADDR workspace,
                                                     GM_ADDR tiling_gm)
{
    SetSyncBaseAddr((unsigned long)sync);

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t*>(tiling_gm);
    int64_t dtype = static_cast<int64_t>(*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf)); // 0-c32; 1-c64
    int64_t n = static_cast<int64_t>(*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + sizeof(uint32_t)));

    int64_t batchSize = static_cast<int64_t>(
        *(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + OFFSET_TWO * sizeof(uint32_t)));
    int64_t blockM = static_cast<int64_t>(
        *(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + OFFSET_THREE * sizeof(uint32_t)));
    int64_t blockN = static_cast<int64_t>(
        *(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + OFFSET_FOUR * sizeof(uint32_t)));
    int64_t tileM = static_cast<int64_t>(
        *(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + OFFSET_FIVE * sizeof(uint32_t)));

    int64_t orgM = n;
    int64_t orgN = n;

    int64_t minMN = orgM > orgN ? orgN : orgM;

    int64_t wEleNum = (minMN + blockN - 1) / blockN * blockN;
    int64_t eyeMatEleNum = ((orgN + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM) *
                           ((orgM + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM * BASE_BLOCK_ELENUM +
                            COL_ALIGNED_ELENUM) *
                           EYE_MATRIX_NUM;

    int64_t dtypeSize = dtype > 0 ? sizeof(float) : sizeof(half);
    // real fp32 (dtype==2): 输入/输出每批 orgM*orgN 个 float (无实虚交错);
    // complex: 每批 orgM*orgN*COMPLEX_ELENUM 个元素
    int64_t batchStride = dtype == 2 ? orgM * orgN * sizeof(float) : orgM * orgN * COMPLEX_ELENUM * dtypeSize;

    /*
     * P2 分流说明: 实数 n<=256 (M<=512) 已改由纯 AIV kernel sgetri_p2_batched
     * 承担 (GetBestKernel 按 dtype==2 && n<=256 分流, 见 sgetri_batched_operation.cpp)。
     * 本 MIX kernel 仅处理: 复数 (dtype 0/1) 与实数 n>256。
     * 曾在 MIX kernel 内做 P2 分支 (AIC 提前 return) —— 实测 aicore timeout:
     * MIX 布局下 kernel 结束协议依赖 AIC 侧完整参与, AIC 提前退出即挂死。
     */

    for (int64_t bid = 0; bid < batchSize; bid++) {
        GM_ADDR cur_W = W + bid * wEleNum * sizeof(uint32_t);
        GM_ADDR cur_eye_gm = eye_gm + bid * eyeMatEleNum * sizeof(float);
        GM_ADDR cur_A_org = A_org + bid * batchStride;
        GM_ADDR cur_A_inv = A_inv + bid * batchStride;

        custom_getri(dtype, orgM, orgN, blockM, blockN, tileM, cur_A_org, cur_A_inv, A_work, cur_W, work_gm, gather1_gm,
                     gather2_gm, gather3_gm, cur_eye_gm, workspace);
    }
}
