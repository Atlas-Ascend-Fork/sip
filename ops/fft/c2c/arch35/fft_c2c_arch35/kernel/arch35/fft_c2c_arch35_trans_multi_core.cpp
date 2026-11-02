/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 收尾转置:[b][x][ỹ] → [b][ỹ][x];v3 组访问(读侧 32B 去放大)
//   in[(b*fftN + x)*fftM + ỹ] → out[(b*fftM + ỹ)*fftN + x](复数单位)

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace {
struct TransPassTilingData {
    int64_t batch;
    int32_t fftN;
    int32_t fftM;
};

constexpr int32_t TRANS_THREAD_NUM = 256;
constexpr int32_t TRANS_GRP = 8;
} // namespace

// 组 g = 输入行 (b,x) × 4 个连续 ỹ:同线程连续读 32B(单扇区),4 次 8B 写落列对齐的
// 4 个输出行;同 ỹ 组相邻 x 的兄弟线程在时间上邻近,合填写侧扇区
__simt_vf__ __launch_bounds__(TRANS_THREAD_NUM) inline void TransGroupSimt(__gm__ float* in, __gm__ float* out,
                                                                           int gBegin, int gCount, int fftN, int fftM)
{
    const int groupsPerRow = (fftM + TRANS_GRP - 1) / TRANS_GRP;
    const int64_t gEnd = static_cast<int64_t>(gBegin) + gCount;
    for (int64_t g = gBegin + threadIdx.x; g < gEnd; g += blockDim.x) {
        const int64_t bx = g / groupsPerRow;
        const int ygrp = static_cast<int>(g - bx * groupsPerRow);
        const int y0 = ygrp * TRANS_GRP;
        const int cnt = (fftM - y0 < TRANS_GRP) ? (fftM - y0) : TRANS_GRP;
        const int64_t b = bx / fftN;
        const int x = static_cast<int>(bx - b * fftN);
        const int64_t inBase = (bx * fftM + y0) * 2;
        const int64_t outBase = ((b * fftM + y0) * fftN + x) * 2;
        for (int j = 0; j < cnt; ++j) {
            const int64_t inOff = inBase + j * 2;
            const int64_t outOff = outBase + static_cast<int64_t>(j) * fftN * 2;
            out[outOff] = in[inOff];
            out[outOff + 1] = in[inOff + 1];
        }
    }
    asc_syncthreads();
}

extern "C" __global__ __vector__ void fft_c2c_arch35_trans_multi_core(__gm__ float* input, __gm__ float* output,
                                                                      __gm__ uint8_t* tiling)
{
    __gm__ TransPassTilingData* td = (__gm__ TransPassTilingData*)tiling;
    const int64_t batch = td->batch;
    const int64_t fftN = td->fftN;
    const int64_t fftM = td->fftM;
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t blockNum = AscendC::GetBlockNum();

    const int64_t groupsPerRow = (fftM + TRANS_GRP - 1) / TRANS_GRP;
    const int64_t totalGroups = batch * fftN * groupsPerRow;
    const int64_t per = (totalGroups + blockNum - 1) / blockNum;
    const int64_t gBegin = blockIdx * per;
    if (gBegin >= totalGroups) {
        return;
    }
    const int64_t gEnd = (gBegin + per) < totalGroups ? (gBegin + per) : totalGroups;

    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<TransGroupSimt>(dim3(TRANS_THREAD_NUM), input, output, static_cast<int>(gBegin),
                                static_cast<int>(gEnd - gBegin), static_cast<int>(fftN), static_cast<int>(fftM));
}
