/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "c_api/asc_simd.h"
#include "kernel_operator.h"
#include "../tiling.h"

namespace FftTranspose {
template <int Tile>
__simd_vf__ inline void ReorderSimd(__ubuf__ float* src, __ubuf__ float* dst, uint16_t rows, uint16_t cols)
{
    namespace Reg = AscendC::Reg;
    Reg::RegTensor<uint32_t> lane, index, component, one, columnIndex;
    Reg::RegTensor<float> value;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), 0);
    Reg::ShiftRights<uint32_t, int16_t>(index, lane, int16_t{1}, full);
    Reg::Muls(index, index, uint32_t{2 * (Tile + 4)}, full);
    Reg::Duplicate(one, uint32_t{1}, full);
    Reg::And(component, lane, one, full);
    Reg::Add(index, index, component, full);
    for (uint16_t c = 0; c < cols; ++c) {
        Reg::Adds(columnIndex, index, static_cast<uint32_t>(2 * c), full);
        for (uint16_t r = 0; r < rows; r += 32) {
            uint32_t count = static_cast<uint32_t>(rows - r) * 2;
            auto mask = Reg::UpdateMask<float>(count);
            Reg::Gather(value, src + 2 * r * (Tile + 4), columnIndex, mask);
            Reg::StoreAlign(dst + 2 * (c * (Tile + 4) + r), value, mask);
        }
    }
}

template <int Tile>
struct TileInfo {
    int64_t input;
    int64_t output;
    int rows;
    int cols;
};

template <int Tile>
__aicore__ inline TileInfo<Tile> Locate(int64_t t, int64_t h, int64_t w)
{
    const int64_t ct = (w + Tile - 1) / Tile;
    const int64_t rt = (h + Tile - 1) / Tile;
    const int64_t perBatch = rt * ct;
    const int64_t b = t / perBatch;
    const int64_t local = t % perBatch;
    const int64_t row = local % rt, col = local / rt;
    const int64_t r = row * Tile, c = col * Tile;
    return {b * h * w + r * w + c, b * h * w + c * h + r, static_cast<int>(h - r < Tile ? h - r : Tile),
            static_cast<int>(w - c < Tile ? w - c : Tile)};
}

template <int Tile>
__aicore__ inline void Load(__ubuf__ float* src, __gm__ float* input, const TileInfo<Tile>& t, int64_t width)
{
    // The final two arguments are GM and UB row pitches in bytes, not gaps.
    asc_copy_gm2ub_align(src, input + 2 * t.input, t.rows, t.cols * 8, 0, 0, false, 0, width * 8, (Tile + 4) * 8);
}

template <int Tile>
__aicore__ inline void Store(__gm__ float* output, __ubuf__ float* dst, const TileInfo<Tile>& t, int64_t height)
{
    asc_copy_ub2gm_align(output + 2 * t.output, dst, t.cols, t.rows * 8, 0, height * 8, (Tile + 4) * 8);
}

template <int Tile, int Slots>
__aicore__ inline void Run(__gm__ float* input, __gm__ float* output, FftTransposeTiling p)
{
    constexpr int floats = Tile * (Tile + 4) * 2;
    __ubuf__ float src[Slots * floats];
    __ubuf__ float dst[Slots * floats];
    const int64_t total = p.batch * ((p.rows + Tile - 1) / Tile) * ((p.cols + Tile - 1) / Tile);
    const int64_t cores = AscendC::GetBlockNum(), id = AscendC::GetBlockIdx();
    const int64_t q = total / cores, rem = total % cores;
    const int64_t count = q + (id < rem ? 1 : 0), start = id * q + (id < rem ? id : rem);
    if (count == 0)
        return;
    for (int s = 0; s < Slots; ++s) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(s));
    }
    // Fill all slots, then consume t while prefetching t+Slots. A slot is
    // reused only after its previous output DMA, including the final drain.
    for (int s = 0; s < Slots && s < count; ++s) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(s));
        Load<Tile>(src + s * floats, input, Locate<Tile>(start + s, p.rows, p.cols), p.cols);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(s));
    }
    for (int64_t i = 0; i < count; ++i) {
        const int s = static_cast<int>(i % Slots);
        const auto ev = static_cast<event_t>(s);
        const auto t = Locate<Tile>(start + i, p.rows, p.cols);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
        ReorderSimd<Tile>(src + s * floats, dst + s * floats, t.rows, t.cols);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev);
        Store<Tile>(output, dst + s * floats, t, p.rows);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
        if (i + Slots < count) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
            Load<Tile>(src + s * floats, input, Locate<Tile>(start + i + Slots, p.rows, p.cols), p.cols);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);
        }
    }
    for (int s = 0; s < Slots; ++s) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(s));
    }
}
} // namespace FftTranspose
