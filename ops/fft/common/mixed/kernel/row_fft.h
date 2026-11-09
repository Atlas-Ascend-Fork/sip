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
__simt_vf__ __launch_bounds__(256) inline void RealBoundary(__ubuf__ float* src, __ubuf__ float* dst,
                                                            __ubuf__ float* roots, int n, bool inverse)
{
    const int m = n / 2;
    const int count = inverse ? m : m + 1;
    for (int k = threadIdx.x; k < count; k += 256) {
        const int a = inverse ? k : k % m, b = inverse ? m - k : (m - k) % m;
        const float ar = src[2 * a], ai = src[2 * a + 1], br = src[2 * b], bi = src[2 * b + 1];
        const float dr = ar - br, di = ai + bi, wr = roots[2 * k], wi = roots[2 * k + 1];
        if (inverse) {
            // Twice the packed inverse spectrum: an unnormalized M inverse
            // then yields the public unnormalized N real inverse.
            dst[2 * k] = ar + br - wr * di - wi * dr;
            dst[2 * k + 1] = ai - bi + wr * dr - wi * di;
        } else {
            dst[2 * k] = 0.5f * (ar + br + wr * di + wi * dr);
            dst[2 * k + 1] = 0.5f * (ai - bi - wr * dr + wi * di);
            if (k == 0 || k == m)
                dst[2 * k + 1] = 0;
        }
    }
}

// Contiguous independent 1D transforms. All stages of a row remain in UB.
template <int Capacity>
__aicore__ inline void Rows(__gm__ float* input, __gm__ float* coeff, __gm__ float* output,
                            const __gm__ AsdSip::OpParam::MixedCommonTiling* t)
{
    const int n = t->n, pass = t->pass;
    const int size = pass == 0 ? n : n / 2;
    __ubuf__ float b0[2 * Capacity + 128], b1[2 * Capacity + 128], roots[2 * Capacity];
    const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t blockIndex = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t quotient = t->rows / blockCount;
    const int64_t remainder = t->rows - quotient * blockCount;
    const int64_t count = quotient + (blockIndex < remainder ? 1 : 0);
    const int64_t start = blockIndex * quotient + (blockIndex < remainder ? blockIndex : remainder);
    asc_copy_gm2ub_align(roots, coeff, 1, n * 8, 0, 0, false, 0, 0, 0);
    const int inFloats = pass == 1 ? n : pass == 2 ? n + 2 : 2 * n;
    const int outFloats = pass == 1 ? n + 2 : pass == 2 ? n : 2 * n;
    for (int row = int(start); row < int(start + count); ++row) {
        asc_copy_gm2ub_align(b0, input + row * inFloats, 1, inFloats * 4, 0, 0, false, 0, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        __ubuf__ float* a = b0;
        __ubuf__ float* b = b1;
        if (pass == 2) {
            asc_vf_call<RealBoundary>(dim3(256), a, b, roots, n, true);
            AscendC::PipeBarrier<PIPE_V>();
            auto* tmp = a;
            a = b;
            b = tmp;
        }
        Radix2Prefix(a, b, roots, uint32_t(size), uint32_t(n), t->radix2Stages, t->inverse != 0);
        AscendC::PipeBarrier<PIPE_V>();
        if (((t->radix2Stages + 1) / 2) & 1) {
            auto* tmp = a;
            a = b;
            b = tmp;
        }
        for (int s = 0; s < t->oddStages; ++s) {
            const int r = t->radix[s], prev = t->prev[s];
            if (r == 3)
                Radix3(a, b, roots, size, n, prev, t->inverse != 0);
            else if (r == 5)
                Radix5(a, b, roots, size, n, prev, t->inverse != 0);
            else if (r == 7)
                Radix7(a, b, roots, size, n, prev, t->inverse != 0);
            else if (t->twoFrequencies)
                asc_vf_call<OddSymmetricStage<true, true>>(dim3(256), a, b, roots, size, n, r, prev);
            else
                asc_vf_call<OddSymmetricStage<false, true>>(dim3(256), a, b, roots, size, n, r, prev);
            AscendC::PipeBarrier<PIPE_V>();
            // The planar odd stage returns in a, unlike the small-radix stages.
            if (r <= 7) {
                auto* tmp = a;
                a = b;
                b = tmp;
            }
        }
        if (pass == 1) {
            asc_vf_call<RealBoundary>(dim3(256), a, b, roots, n, false);
            AscendC::PipeBarrier<PIPE_V>();
            a = b;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        asc_copy_ub2gm_align(output + row * outFloats, a, 1, outFloats * 4, 0, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}
} // namespace fft_mixed
