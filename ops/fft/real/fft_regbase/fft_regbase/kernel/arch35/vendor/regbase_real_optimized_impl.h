/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FFT_REAL_OPTIMIZED_IMPL_H
#define FFT_REAL_OPTIMIZED_IMPL_H

namespace real_optimized {

// One lane owns BOTH k and M-k. Full chunks may use contiguous loads on
// the aligned low side; the reverse high side deliberately remains masked
// Gather/Scatter. No unmasked access is used for the first partial chunk.
template <int N, bool C2R, bool Contiguous>
__simd_callee__ inline void ConvertPairBlock(__ubuf__ float* src, __ubuf__ float* tw, __ubuf__ float* dst,
                                             uint32_t first, uint32_t count)
{
    using namespace AscendC;
    Reg::MaskReg mask = Reg::UpdateMask<float>(count);
    Reg::RegTensor<uint32_t> k, index, mirror, limit;
    Reg::RegTensor<float> ar, ai, br, bi, wr, wi;
    Reg::RegTensor<float> er, ei, dr, di, tr, ti, yr, yi, temporary;
    Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(k), static_cast<int32_t>(first));
    Reg::Muls(index, k, uint32_t{2}, mask);
    if constexpr (Contiguous) {
        // Caller guarantees 64 active lanes, first%64==0, and an interior
        // range entirely below N/4. Both 512-byte loads are in bounds.
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, src + 2U * first);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(wr, wi, tw + 2U * first);
    } else {
        Reg::Gather(ar, src, index, mask);
        Reg::Gather(wr, tw, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(ai, src, index, mask);
        Reg::Gather(wi, tw, index, mask);
    }
    Reg::Duplicate(limit, uint32_t{N / 2}, mask);
    Reg::Sub(mirror, limit, k, mask);
    Reg::Muls(mirror, mirror, uint32_t{2}, mask);
    Reg::Gather(br, src, mirror, mask);
    Reg::Adds(index, mirror, uint32_t{1}, mask);
    Reg::Gather(bi, src, index, mask);
    Reg::Add(er, ar, br, mask);
    Reg::Sub(ei, ai, bi, mask);
    Reg::Sub(dr, ar, br, mask);
    Reg::Add(di, ai, bi, mask);
    Reg::Mul(tr, wr, di, mask);
    Reg::Mul(temporary, wi, dr, mask);
    Reg::Add(tr, tr, temporary, mask);
    Reg::Mul(ti, wi, di, mask);
    Reg::Mul(temporary, wr, dr, mask);
    Reg::Sub(ti, ti, temporary, mask);
    if constexpr (C2R) {
        Reg::Sub(yr, er, tr, mask);
        Reg::Sub(yi, ei, ti, mask);
    } else {
        Reg::Add(yr, er, tr, mask);
        Reg::Add(yi, ei, ti, mask);
        Reg::Muls(yr, yr, 0.5f, mask);
        Reg::Muls(yi, yi, 0.5f, mask);
    }
    if constexpr (Contiguous) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(dst + 2U * first, yr, yi, mask);
    } else {
        Reg::Muls(index, k, uint32_t{2}, mask);
        Reg::Scatter(dst, yr, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(dst, yi, index, mask);
    }
    if constexpr (C2R) {
        Reg::Add(yr, er, tr, mask);
        Reg::Add(yi, ei, ti, mask);
        Reg::Muls(yi, yi, -1.0f, mask);
    } else {
        Reg::Sub(yr, er, tr, mask);
        Reg::Sub(yi, ti, ei, mask);
        Reg::Muls(yr, yr, 0.5f, mask);
        Reg::Muls(yi, yi, 0.5f, mask);
    }
    Reg::Scatter(dst, yr, mirror, mask);
    Reg::Adds(mirror, mirror, uint32_t{1}, mask);
    Reg::Scatter(dst, yi, mirror, mask);
}

template <int N, bool C2R>
__simd_callee__ inline void ConvertSpecialPoints(__ubuf__ float* src, __ubuf__ float* dst)
{
    using namespace AscendC;
    uint32_t one = 1;
    Reg::MaskReg mask = Reg::UpdateMask<float>(one);
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<float> a, b, sum, difference, zero;
    Reg::Duplicate(index, uint32_t{0}, mask);
    Reg::Gather(a, src, index, mask);
    Reg::Duplicate(index, uint32_t{C2R ? N : 1}, mask);
    Reg::Gather(b, src, index, mask);
    Reg::Add(sum, a, b, mask);
    Reg::Sub(difference, a, b, mask);
    Reg::Duplicate(index, uint32_t{0}, mask);
    Reg::Scatter(dst, sum, index, mask);
    Reg::Duplicate(index, uint32_t{C2R ? 1 : N}, mask);
    Reg::Scatter(dst, difference, index, mask);
    if constexpr (!C2R) {
        Reg::Duplicate(zero, 0.0f, mask);
        Reg::Duplicate(index, uint32_t{1}, mask);
        Reg::Scatter(dst, zero, index, mask);
        Reg::Duplicate(index, uint32_t{N + 1}, mask);
        Reg::Scatter(dst, zero, index, mask);
    }
    // k=M/2 is its own partner. Emit it once, with the exact +/-i twiddle.
    Reg::Duplicate(index, uint32_t{N / 2}, mask);
    Reg::Gather(a, src, index, mask);
    Reg::Adds(index, index, uint32_t{1}, mask);
    Reg::Gather(b, src, index, mask);
    if constexpr (C2R)
        Reg::Muls(a, a, 2.0f, mask);
    Reg::Muls(b, b, C2R ? -2.0f : -1.0f, mask);
    Reg::Scatter(dst, b, index, mask);
    Reg::Duplicate(index, uint32_t{N / 2}, mask);
    Reg::Scatter(dst, a, index, mask);
}

template <int N, bool C2R, bool Contiguous>
__no_simd_vf_fusion__ __simd_vf__ inline void ConvertRows(__ubuf__ float* input, __ubuf__ float* tw,
                                                          __ubuf__ float* output, uint16_t tileBatch)
{
    constexpr uint32_t quarter = N / 4;
    constexpr uint32_t firstCount = quarter < 64 ? quarter - 1 : 63;
#pragma unroll 1
    for (uint16_t row = 0; row < tileBatch; ++row) {
        __ubuf__ float* src = input + row * (C2R ? N + 8 : N);
        __ubuf__ float* dst = output + row * (C2R ? N : N + 8);
        ConvertPairBlock<N, C2R, false>(src, tw, dst, 1, firstCount);
#pragma unroll 1
        for (uint16_t block = 1; block < quarter / 64; ++block)
            ConvertPairBlock<N, C2R, Contiguous>(src, tw, dst, uint32_t(block) * 64U, 64U);
        ConvertSpecialPoints<N, C2R>(src, dst);
    }
}

template <int N, bool C2R>
__aicore__ inline void LoadTile(__ubuf__ float* packed, __ubuf__ float* half, __gm__ float* input, int64_t globalBatch,
                                uint16_t count)
{
    if constexpr (C2R)
        asc_copy_gm2ub_align(half, input + globalBatch * (N + 2), count, (N + 2) * 4, 0, 0, false, 0, (N + 2) * 4,
                             (N + 8) * 4);
    else
        asc_copy_gm2ub_align(packed, input + globalBatch * N, 1, count * N * 4, 0, 0, false, 0, 0, 0);
}

template <int N, bool C2R, int Config, bool DoubleBuffer>
__aicore__ inline void Run(__gm__ float* input, __gm__ float* coreTwiddles, __gm__ float* boundaryTwiddles,
                           __gm__ float* output, int64_t batch)
{
    static_assert(N >= 32 && N <= 8192 && (N & (N - 1)) == 0);
    static_assert(Config >= 1 && Config <= 6);
    static_assert(Config != 5 || !DoubleBuffer);
    constexpr int m = N / 2;
    constexpr int baseCapacity = m <= 256 ? 256 / m : (m == 512 ? 2 : 1);
    constexpr int tileCapacity = Config == 4 || (Config == 6 && N == 512) ? 4 :
                                 Config == 3 || Config == 5               ? 2 :
                                                                            baseCapacity;
    constexpr int packedFloats = m >= 512 ? tileCapacity * (N + 8) + 128 : tileCapacity * N;
    constexpr int halfFloats = m >= 512 ? packedFloats : tileCapacity * (N + 8);
    constexpr int slots = DoubleBuffer ? 2 : 1;
    constexpr int realTwFloats = N / 2 + 8;
    static_assert(4 * (slots * (packedFloats + halfFloats) + N + realTwFloats) <= 253952);
    __ubuf__ float packedStorage[slots * packedFloats];
    __ubuf__ float halfStorage[slots * halfFloats];
    __ubuf__ float fftTw[N];
    __ubuf__ float realTw[realTwFloats];
    const CoreBatchRange range = GetCoreBatchRange(batch);
    if (!range.count)
        return;
    asc_copy_gm2ub_align(fftTw, coreTwiddles, 1, N * 4, 0, 0, false, 0, 0, 0);
    asc_copy_gm2ub_align(realTw, boundaryTwiddles, 1, realTwFloats * 4, 0, 0, false, 0, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    if constexpr (DoubleBuffer) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        const uint16_t count = static_cast<uint16_t>(range.count < tileCapacity ? range.count : tileCapacity);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        LoadTile<N, C2R>(packedStorage, halfStorage, input, range.start, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
    int64_t tileIndex = 0;
    for (int64_t local = 0; local < range.count; local += tileCapacity, ++tileIndex) {
        const int64_t left = range.count - local;
        const uint16_t count = static_cast<uint16_t>(left < tileCapacity ? left : tileCapacity);
        const int64_t globalBatch = range.start + local;
        const int slot = DoubleBuffer ? (tileIndex & 1) : 0;
        const int eventId = slot ? EVENT_ID1 : EVENT_ID0;
        __ubuf__ float* packed = packedStorage + slot * packedFloats;
        __ubuf__ float* half = halfStorage + slot * halfFloats;
        if constexpr (DoubleBuffer) {
            const int64_t next = local + tileCapacity;
            if (next < range.count) {
                const int nextSlot = 1 - slot;
                const int nextEvent = nextSlot ? EVENT_ID1 : EVENT_ID0;
                const int64_t nextLeft = range.count - next;
                const uint16_t nextCount = static_cast<uint16_t>(nextLeft < tileCapacity ? nextLeft : tileCapacity);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(nextEvent);
                LoadTile<N, C2R>(packedStorage + nextSlot * packedFloats, halfStorage + nextSlot * halfFloats, input,
                                 range.start + next, nextCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextEvent);
            }
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            LoadTile<N, C2R>(packed, half, input, globalBatch, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        if constexpr (C2R) {
            ConvertRows<N, true, Config != 1>(half, realTw, packed, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        __ubuf__ float* fftOutput = packed;
        __ubuf__ float* halfOutput = half;
        if constexpr (m <= 64) {
            StockhamRadix2FusedRegisterDirectVf<m>(packed, fftTw, packed, count);
        } else if constexpr (m == 128) {
            StockhamRadix2FusedRegisterLazy128DirectVf<Config == 6>(packed, fftTw, packed, count);
        } else if constexpr (m == 256) {
            StockhamRadix2FusedRegisterLazy256DirectVf<Config == 6>(packed, fftTw, packed, count);
        } else {
            StockhamRadix4FusedUnrolledDirectVf<m, C2R>(packed, fftTw, half, count);
            // kRadix4Passes=ceil(log2(m)/2) already includes the optional radix-2 tail.
            // This matches the vendor FinalBuffer<m, FusedRadix4UnrolledDirect> helper.
            if constexpr ((FftTraits<m>::kRadix4Passes & 1) != 0) {
                fftOutput = half;
                halfOutput = packed;
            }
        }
        if constexpr (!C2R) {
            AscendC::PipeBarrier<PIPE_V>();
            ConvertRows<N, false, Config != 1>(fftOutput, realTw, halfOutput, count);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
        if constexpr (C2R)
            asc_copy_ub2gm_align(output + globalBatch * N, fftOutput, 1, count * N * 4, 0, 0, 0);
        else
            asc_copy_ub2gm_align(output + globalBatch * (N + 2), halfOutput, count, (N + 2) * 4, 0, (N + 2) * 4,
                                 (N + 8) * 4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    if constexpr (DoubleBuffer)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}
} // namespace real_optimized
#endif
