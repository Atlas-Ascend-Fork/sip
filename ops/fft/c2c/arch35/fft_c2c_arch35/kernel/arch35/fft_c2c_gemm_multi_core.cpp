/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GEMM 短维 pass 设备核:变换连续维 fftN

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace AscendC;

namespace {
// 与 tiling/fft_c2c_arch35_gemm_tiling.h 的 FftGemmTilingData 逐字段一致
// (核侧本地定义,与 mix kernel 的 FftAllMixTilingData 同惯例)
struct GemmPassTilingData {
    int64_t rows;
    int32_t kDim;
    int32_t nDim;
    int32_t isInverse;
    uint32_t coreNum;
};

using GemmArchTag = Arch::Ascend950;
using GemmPolicy = Gemm::MmadPingpong<GemmArchTag, /*enableUnitFlag*/ false, /*useHF32*/ false, /*l0CStages*/ 2,
                                      /*enableL1Resident*/ true, /*l1AStages*/ 2, /*l1BStages*/ 1, /*l0AStages*/ 2,
                                      /*l0BStages*/ 2>;
using GemmL1Tile = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
using GemmL0Tile = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<32>>;
using GemmBlockMmad = Gemm::Block::BlockMmadTla<GemmPolicy, GemmL1Tile, GemmL0Tile, float, float, float>;

constexpr int32_t GEMM_MTILE = 128; // 首版固定(定标:C_call≈0,MT 64/128/256 等价);L0C 上限 256
} // namespace

extern "C" CATLASS_GLOBAL void fft_c2c_arch35_gemm_multi_core(GM_ADDR a, GM_ADDR w, GM_ADDR c, GM_ADDR tiling)
{
    __gm__ GemmPassTilingData* td = (__gm__ GemmPassTilingData*)tiling;
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= td->coreNum) {
        return;
    }

    Arch::Resource<GemmArchTag> resource;
    GemmBlockMmad blockMmad(resource, 0);

    const int64_t rows = td->rows;
    const int64_t kDim = td->kDim;
    const int64_t nDim = td->nDim;

    // 平衡切分(与 bench/FFT89_CORE_MAT_RANGE 同式)
    const int64_t per = rows / td->coreNum;
    const int64_t rem = rows % td->coreNum;
    const int64_t mStart = blockIdx * per + (blockIdx < rem ? blockIdx : rem);
    const int64_t mEnd = mStart + per + (blockIdx < rem ? 1 : 0);

    GlobalTensor<float> gA, gW, gC;
    gA.SetGlobalBuffer((__gm__ float*)a);
    gW.SetGlobalBuffer((__gm__ float*)w);
    gC.SetGlobalBuffer((__gm__ float*)c);

    auto layoutW = tla::MakeLayout<float, layout::RowMajor>(static_cast<uint32_t>(kDim), static_cast<uint32_t>(nDim));
    auto tensorW = tla::MakeTensor(gW, layoutW, Arch::PositionGM{});
    auto layoutC = tla::MakeLayout<float, layout::RowMajor>(static_cast<uint32_t>(rows), static_cast<uint32_t>(nDim));
    auto tensorC = tla::MakeTensor(gC, layoutC, Arch::PositionGM{});

    for (int64_t m0 = mStart; m0 < mEnd; m0 += GEMM_MTILE) {
        const uint32_t mt = static_cast<uint32_t>(mEnd - m0 < static_cast<int64_t>(GEMM_MTILE) ? (mEnd - m0) :
                                                                                                 GEMM_MTILE);
        GlobalTensor<float> gATile;
        gATile.SetGlobalBuffer((__gm__ float*)a + m0 * kDim);
        auto layoutA = tla::MakeLayout<float, layout::RowMajor>(mt, static_cast<uint32_t>(kDim));
        auto tensorA = tla::MakeTensor(gATile, layoutA, Arch::PositionGM{});

        auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0u, 0u), tla::MakeShape(mt, static_cast<uint32_t>(kDim)));
        auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0u, 0u),
                                    tla::MakeShape(static_cast<uint32_t>(kDim), static_cast<uint32_t>(nDim)));
        auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(static_cast<uint32_t>(m0), 0u),
                                    tla::MakeShape(mt, static_cast<uint32_t>(nDim)));

        blockMmad(tensorBlockA, tensorBlockW, tensorBlockC,
                  GemmCoord(mt, static_cast<uint32_t>(nDim), static_cast<uint32_t>(kDim)));
    }
}
