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
#include "simt_api/asc_simt.h"
#include "params/fft_mixed_common.h"
#include "radix3.h"
#include "radix4.h"
#include "radix57.h"
#include "odd_symmetric.h"
#include "batched_vf.h"
#include "tile_simd.h"
namespace fft_mixed {
__simd_callee__ inline void Radix2Body(__ubuf__ float* src, __ubuf__ float* dst, __ubuf__ float* roots, uint32_t size,
                                       uint32_t rootN, uint32_t half)
{
    using namespace AscendC;
    const uint32_t part = size / 2;
    for (uint32_t first = 0; first < part; first += 64) {
        uint32_t active = part - first < 64 ? part - first : 64;
        Reg::MaskReg mask = Reg::UpdateMask<float>(active);
        Reg::RegTensor<uint32_t> q, j, halfMask, index, output0;
        Reg::RegTensor<float> ur, ui, vr, vi, wr, wi, tr, ti, temporary;
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(q), int32_t(first));
        Reg::Duplicate(halfMask, half - 1U, mask);
        Reg::And(j, q, halfMask, mask);
        Reg::Muls(index, q, uint32_t{2}, mask);
        Reg::Gather(ur, src, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(ui, src, index, mask);
        Reg::Adds(index, index, size - 1U, mask);
        Reg::Gather(vr, src, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(vi, src, index, mask);
        // roots contains all N roots, not the compact radix-2 twiddle layout.
        Reg::Muls(index, j, 2U * (rootN / (2U * half)), mask);
        Reg::Gather(wr, roots, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(wi, roots, index, mask);
        Reg::Mul(tr, vr, wr, mask);
        Reg::Mul(temporary, vi, wi, mask);
        Reg::Sub(tr, tr, temporary, mask);
        Reg::Mul(ti, vr, wi, mask);
        Reg::Mul(temporary, vi, wr, mask);
        Reg::Add(ti, ti, temporary, mask);
        Reg::Sub(vr, ur, tr, mask);
        Reg::Sub(vi, ui, ti, mask);
        Reg::Add(ur, ur, tr, mask);
        Reg::Add(ui, ui, ti, mask);
        Reg::Muls(output0, q, uint32_t{2}, mask);
        Reg::Sub(output0, output0, j, mask);
        Reg::Muls(index, output0, uint32_t{2}, mask);
        Reg::Scatter(dst, ur, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(dst, ui, index, mask);
        Reg::Adds(index, index, 2U * half - 1U, mask);
        Reg::Scatter(dst, vr, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(dst, vi, index, mask);
    }
}
__no_simd_vf_fusion__ __simd_vf__ inline void Radix2Prefix(__ubuf__ float* src, __ubuf__ float* dst,
                                                           __ubuf__ float* roots, uint32_t size, uint32_t rootN,
                                                           int stages, bool inverse)
{
    uint32_t half = 1;
    for (int stage = 0; stage < stages; ++stage) {
        if (stage + 1 < stages) {
            Radix4(src, dst, roots, size, rootN, half, inverse);
            ++stage;
            half *= 2;
        } else {
            Radix2Body(src, dst, roots, size, rootN, half);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        auto* tmp = src;
        src = dst;
        dst = tmp;
        half *= 2;
    }
}
template <AscendC::HardEvent Event>
__aicore__ inline void SetRowSlot(int slot)
{
    if (slot == 0)
        AscendC::SetFlag<Event>(EVENT_ID0);
    else
        AscendC::SetFlag<Event>(EVENT_ID1);
}
template <AscendC::HardEvent Event>
__aicore__ inline void WaitRowSlot(int slot)
{
    if (slot == 0)
        AscendC::WaitFlag<Event>(EVENT_ID0);
    else
        AscendC::WaitFlag<Event>(EVENT_ID1);
}
// Preserve the specialized small-radix formulas; larger prime stages retain
// planar symmetry and single/two-frequency reuse, all over the actual tile rows.
__aicore__ inline __ubuf__ float* ComputeTileButterfly(__ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* roots,
                                                       const __gm__ AsdSip::OpParam::MixedCommonTiling* t, int rows,
                                                       int stride)
{
    const int n = t->n, pass = t->pass, size = pass == 0 ? n : n / 2;
    if (pass == 2) {
        asc_vf_call<BatchedRealBoundary>(dim3(256), a, b, roots, n, true, rows, stride);
        AscendC::PipeBarrier<PIPE_V>();
        auto* tmp = a;
        a = b;
        b = tmp;
    }
    TilePrefix(a, b, roots, uint32_t(size), uint32_t(n), t->radix2Stages, t->inverse != 0, uint32_t(rows),
               uint32_t(stride));
    AscendC::PipeBarrier<PIPE_V>();
    if (((t->radix2Stages + 1) / 2) & 1) {
        auto* tmp = a;
        a = b;
        b = tmp;
    }
    for (int s = 0; s < t->oddStages; ++s) {
        const int radix = t->radix[s], prev = t->prev[s];
        if (radix == 3)
            TileRadix3(a, b, roots, size, n, prev, t->inverse != 0, rows, stride);
        else if (radix == 5)
            TileRadix5(a, b, roots, size, n, prev, t->inverse != 0, rows, stride);
        else if (radix == 7)
            TileRadix7(a, b, roots, size, n, prev, t->inverse != 0, rows, stride);
        else if (t->twoFrequencies)
            asc_vf_call<BatchedOddStage<true, true>>(dim3(256), a, b, roots, size, n, radix, prev, rows, stride);
        else
            asc_vf_call<BatchedOddStage<false, true>>(dim3(256), a, b, roots, size, n, radix, prev, rows, stride);
        AscendC::PipeBarrier<PIPE_V>();
        if (radix <= 7) {
            auto* tmp = a;
            a = b;
            b = tmp;
        }
    }
    if (pass == 1) {
        asc_vf_call<BatchedRealBoundary>(dim3(256), a, b, roots, n, false, rows, stride);
        AscendC::PipeBarrier<PIPE_V>();
        a = b;
    }
    return a;
}
// A single complex row needs no cross-row index arithmetic. Keep its original
// dedicated butterflies while the caller retains dynamic UB and DMA scheduling.
__aicore__ inline __ubuf__ float* ComputeSingleComplexRow(__ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* roots,
                                                          const __gm__ AsdSip::OpParam::MixedCommonTiling* t)
{
    const int n = t->n;
    Radix2Prefix(a, b, roots, uint32_t(n), uint32_t(n), t->radix2Stages, t->inverse != 0);
    AscendC::PipeBarrier<PIPE_V>();
    if (((t->radix2Stages + 1) / 2) & 1) {
        auto* temp = a;
        a = b;
        b = temp;
    }
    for (int stage = 0; stage < t->oddStages; ++stage) {
        const int radix = t->radix[stage], prev = t->prev[stage];
        if (radix == 3)
            Radix3(a, b, roots, n, n, prev, t->inverse != 0);
        else if (radix == 5)
            Radix5(a, b, roots, n, n, prev, t->inverse != 0);
        else if (radix == 7)
            Radix7(a, b, roots, n, n, prev, t->inverse != 0);
        else if (t->twoFrequencies)
            asc_vf_call<OddSymmetricStage<true, true>>(dim3(256), a, b, roots, n, n, radix, prev);
        else
            asc_vf_call<OddSymmetricStage<false, true>>(dim3(256), a, b, roots, n, n, radix, prev);
        AscendC::PipeBarrier<PIPE_V>();
        if (radix <= 7) {
            auto* temp = a;
            a = b;
            b = temp;
        }
    }
    return a;
}
__aicore__ inline void GetTileRowRange(const __gm__ AsdSip::OpParam::MixedCommonTiling* t, int& start, int& count)
{
    const int blocks = AscendC::GetBlockNum(), block = AscendC::GetBlockIdx();
    const int group = t->rowGroup, groups = (t->rows - 1) / group + 1;
    const int q = groups / blocks, rem = groups % blocks;
    start = (block * q + (block < rem ? block : rem)) * group;
    const int assigned = (q + (block < rem)) * group;
    count = assigned < t->rows - start ? assigned : t->rows - start;
}
template <bool CompactReal = false, int Slots = 2>
__aicore__ inline void RowsTileAdaptive(__gm__ float* input, __gm__ float* coeff, __gm__ float* output,
                                        const __gm__ AsdSip::OpParam::MixedCommonTiling* t)
{
    static_assert(Slots == 1 || Slots == 2);
    static_assert(Slots == 2 || !CompactReal);
    // Slots == 1 is key 26 only: the host selects complex N >= 6752, where tileRows is exactly one.
    // The 212 KiB arena reserves exactly
    // 2*N root floats; each of the data regions is 32-byte aligned.
    constexpr int ArenaFloats = 212 * 1024 / sizeof(float);
    __ubuf__ float arena[ArenaFloats];
    const int n = t->n, pass = t->pass;
    // Real FFT uses N/2 complex values. N+128 also covers the N+2 half spectrum.
    const int stride = ((CompactReal ? n : 2 * n) + 135) / 8 * 8;
    const int SlotFloats = ((ArenaFloats - 2 * n) / (2 * Slots) / 8) * 8;
    __ubuf__ float* a = arena;
    __ubuf__ float* b = arena + Slots * SlotFloats;
    __ubuf__ float* roots = arena + 2 * Slots * SlotFloats;
    const int tileRows = SlotFloats / stride;
    int start, count;
    GetTileRowRange(t, start, count);
    const int inFloats = pass == 1 ? n : pass == 2 ? n + 2 : 2 * n;
    const int outFloats = pass == 1 ? n + 2 : pass == 2 ? n : 2 * n;
    const int tiles = (count + tileRows - 1) / tileRows;
    asc_copy_gm2ub_align(roots, coeff, 1, n * 8, 0, 0, false, 0, 0, 0);
    const int firstRows = count < tileRows ? count : tileRows;
    asc_copy_gm2ub_align(a, input + start * inFloats, firstRows, inFloats * 4, 0, 0, false, 0, inFloats * 4,
                         stride * 4);
    SetRowSlot<AscendC::HardEvent::MTE2_V>(0);
    for (int tile = 0; tile < tiles; ++tile) {
        const int slot = tile % Slots, next = (tile + 1) % Slots;
        if constexpr (Slots == 2) {
            if (tile + 1 < tiles) {
                if (tile > 0)
                    WaitRowSlot<AscendC::HardEvent::MTE3_MTE2>(next);
                const int left = count - (tile + 1) * tileRows;
                const int rows = left < tileRows ? left : tileRows;
                asc_copy_gm2ub_align(a + next * SlotFloats, input + (start + (tile + 1) * tileRows) * inFloats, rows,
                                     inFloats * 4, 0, 0, false, 0, inFloats * 4, stride * 4);
                SetRowSlot<AscendC::HardEvent::MTE2_V>(next);
            }
        }
        WaitRowSlot<AscendC::HardEvent::MTE2_V>(slot);
        const int left = count - tile * tileRows, rows = left < tileRows ? left : tileRows;
        __ubuf__ float* result;
        if constexpr (Slots == 1)
            result = ComputeSingleComplexRow(a, b, roots, t);
        else
            result = ComputeTileButterfly(a + slot * SlotFloats, b + slot * SlotFloats, roots, t, rows, stride);
        SetRowSlot<AscendC::HardEvent::V_MTE3>(slot);
        WaitRowSlot<AscendC::HardEvent::V_MTE3>(slot);
        asc_copy_ub2gm_align(output + (start + tile * tileRows) * outFloats, result, rows, outFloats * 4, 0,
                             outFloats * 4, stride * 4);
        SetRowSlot<AscendC::HardEvent::MTE3_MTE2>(slot);
        if constexpr (Slots == 1) {
            WaitRowSlot<AscendC::HardEvent::MTE3_MTE2>(slot);
            if (tile + 1 < tiles) {
                const int remain = count - (tile + 1) * tileRows;
                const int nextRows = remain < tileRows ? remain : tileRows;
                asc_copy_gm2ub_align(a, input + (start + (tile + 1) * tileRows) * inFloats, nextRows, inFloats * 4, 0,
                                     0, false, 0, inFloats * 4, stride * 4);
                SetRowSlot<AscendC::HardEvent::MTE2_V>(0);
            }
        }
    }
    if constexpr (Slots == 2) {
        WaitRowSlot<AscendC::HardEvent::MTE3_MTE2>(0);
        if (tiles > 1)
            WaitRowSlot<AscendC::HardEvent::MTE3_MTE2>(1);
    }
}
// Three complete tile buffers rotate through input, compute scratch and output.
// The previous output must finish MTE3 before that buffer is reused by MTE2.
template <bool CompactReal>
__aicore__ inline void RowsTileThree(__gm__ float* input, __gm__ float* coeff, __gm__ float* output,
                                     const __gm__ AsdSip::OpParam::MixedCommonTiling* t)
{
    constexpr int ArenaFloats = 212 * 1024 / sizeof(float);
    __ubuf__ float arena[ArenaFloats];
    const int n = t->n, stride = ((CompactReal ? n : 2 * n) + 135) / 8 * 8;
    const int inFloats = t->pass == 0 ? 2 * n : t->pass == 1 ? n : n + 2;
    const int outFloats = t->pass == 0 ? 2 * n : t->pass == 1 ? n + 2 : n;
    const int capacity = ((ArenaFloats - 2 * n) / 3 / 8) * 8;
    const int tileRows = capacity / stride;
    __ubuf__ float* current = arena;
    __ubuf__ float* scratch = arena + capacity;
    __ubuf__ float* spare = arena + 2 * capacity;
    __ubuf__ float* roots = arena + 3 * capacity;
    int start, count;
    GetTileRowRange(t, start, count);
    const int tiles = (count + tileRows - 1) / tileRows;
    asc_copy_gm2ub_align(roots, coeff, 1, n * 8, 0, 0, false, 0, 0, 0);
    const int firstRows = count < tileRows ? count : tileRows;
    asc_copy_gm2ub_align(current, input + start * inFloats, firstRows, inFloats * 4, 0, 0, false, 0, inFloats * 4,
                         stride * 4);
    SetRowSlot<AscendC::HardEvent::MTE2_V>(0);
    for (int tile = 0; tile < tiles; ++tile) {
        // Consume every output-completion flag, including the penultimate one
        // when the last tile has no further input to prefetch.
        if (tile > 0)
            WaitRowSlot<AscendC::HardEvent::MTE3_MTE2>(0);
        if (tile + 1 < tiles) {
            const int left = count - (tile + 1) * tileRows;
            const int rows = left < tileRows ? left : tileRows;
            asc_copy_gm2ub_align(spare, input + (start + (tile + 1) * tileRows) * inFloats, rows, inFloats * 4, 0, 0,
                                 false, 0, inFloats * 4, stride * 4);
            SetRowSlot<AscendC::HardEvent::MTE2_V>((tile + 1) % 2);
        }
        WaitRowSlot<AscendC::HardEvent::MTE2_V>(tile % 2);
        const int left = count - tile * tileRows, rows = left < tileRows ? left : tileRows;
        __ubuf__ float* result = ComputeTileButterfly(current, scratch, roots, t, rows, stride);
        SetRowSlot<AscendC::HardEvent::V_MTE3>(0);
        WaitRowSlot<AscendC::HardEvent::V_MTE3>(0);
        asc_copy_ub2gm_align(output + (start + tile * tileRows) * outFloats, result, rows, outFloats * 4, 0,
                             outFloats * 4, stride * 4);
        SetRowSlot<AscendC::HardEvent::MTE3_MTE2>(0);
        // Keep the output buffer unavailable to the next compute; the other
        // compute buffer becomes scratch. The already-prefetched spare is input.
        __ubuf__ float* freeBuffer = result == current ? scratch : current;
        current = spare;
        scratch = freeBuffer;
        spare = result;
    }
    WaitRowSlot<AscendC::HardEvent::MTE3_MTE2>(0);
}
} // namespace fft_mixed
