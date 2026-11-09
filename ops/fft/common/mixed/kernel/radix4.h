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
namespace fft_mixed {
// q spans butterflies across blocks, so early stages keep all SIMD lanes useful.
__simd_callee__ inline void Radix4(__ubuf__ float* src, __ubuf__ float* dst, __ubuf__ float* roots, uint32_t size,
                                   uint32_t rootN, uint32_t prev, bool inverse)
{
    using namespace AscendC;
    const uint32_t part = size / 4U, step = rootN / (4U * prev);
    const float sign = inverse ? 1.0f : -1.0f;
    for (uint32_t first = 0; first < part; first += 64U) {
        uint32_t active = part - first < 64U ? part - first : 64U;
        Reg::MaskReg mask = Reg::UpdateMask<float>(active);
        Reg::RegTensor<uint32_t> q, j, bits, index, output;
        Reg::RegTensor<float> ar, ai, br, bi, cr, ci, dr, di;
        Reg::RegTensor<float> wr, wi, real, temp, ur, ui, vr, vi;
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(q), int32_t(first));
        Reg::Duplicate(bits, prev - 1U, mask);
        Reg::And(j, q, bits, mask);
        Reg::Muls(index, q, uint32_t{2}, mask);
        Reg::Adds(index, index, 0U * part, mask);
        Reg::Gather(ar, src, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(ai, src, index, mask);
        Reg::Muls(index, q, uint32_t{2}, mask);
        Reg::Adds(index, index, 2U * part, mask);
        Reg::Gather(br, src, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(bi, src, index, mask);
        if (prev != 1U) {
            Reg::Muls(index, j, 2U * step, mask);
            Reg::Gather(wr, roots, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(wi, roots, index, mask);
            Reg::Mul(real, br, wr, mask);
            Reg::Mul(temp, bi, wi, mask);
            Reg::Sub(real, real, temp, mask);
            Reg::Mul(temp, br, wi, mask);
            Reg::Mul(bi, bi, wr, mask);
            Reg::Add(bi, bi, temp, mask);
            Reg::Adds(br, real, 0.0f, mask);
        }
        Reg::Muls(index, q, uint32_t{2}, mask);
        Reg::Adds(index, index, 4U * part, mask);
        Reg::Gather(cr, src, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(ci, src, index, mask);
        if (prev != 1U) {
            Reg::Muls(index, j, 4U * step, mask);
            Reg::Gather(wr, roots, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(wi, roots, index, mask);
            Reg::Mul(real, cr, wr, mask);
            Reg::Mul(temp, ci, wi, mask);
            Reg::Sub(real, real, temp, mask);
            Reg::Mul(temp, cr, wi, mask);
            Reg::Mul(ci, ci, wr, mask);
            Reg::Add(ci, ci, temp, mask);
            Reg::Adds(cr, real, 0.0f, mask);
        }
        Reg::Muls(index, q, uint32_t{2}, mask);
        Reg::Adds(index, index, 6U * part, mask);
        Reg::Gather(dr, src, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(di, src, index, mask);
        if (prev != 1U) {
            Reg::Muls(index, j, 6U * step, mask);
            Reg::Gather(wr, roots, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(wi, roots, index, mask);
            Reg::Mul(real, dr, wr, mask);
            Reg::Mul(temp, di, wi, mask);
            Reg::Sub(real, real, temp, mask);
            Reg::Mul(temp, dr, wi, mask);
            Reg::Mul(di, di, wr, mask);
            Reg::Add(di, di, temp, mask);
            Reg::Adds(dr, real, 0.0f, mask);
        }
        Reg::Add(ur, ar, cr, mask);
        Reg::Add(ui, ai, ci, mask);
        Reg::Sub(ar, ar, cr, mask);
        Reg::Sub(ai, ai, ci, mask);
        Reg::Add(vr, br, dr, mask);
        Reg::Add(vi, bi, di, mask);
        Reg::Sub(br, br, dr, mask);
        Reg::Sub(bi, bi, di, mask);
        Reg::Muls(br, br, sign, mask);
        Reg::Muls(bi, bi, sign, mask);
        Reg::Muls(output, q, uint32_t{4}, mask);
        Reg::Muls(index, j, uint32_t{3}, mask);
        Reg::Sub(output, output, index, mask);
        Reg::Add(cr, ur, vr, mask);
        Reg::Add(ci, ui, vi, mask);
        Reg::Muls(index, output, uint32_t{2}, mask);
        Reg::Adds(index, index, 0U * prev, mask);
        Reg::Scatter(dst, cr, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(dst, ci, index, mask);
        Reg::Sub(cr, ar, bi, mask);
        Reg::Add(ci, ai, br, mask);
        Reg::Muls(index, output, uint32_t{2}, mask);
        Reg::Adds(index, index, 2U * prev, mask);
        Reg::Scatter(dst, cr, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(dst, ci, index, mask);
        Reg::Sub(cr, ur, vr, mask);
        Reg::Sub(ci, ui, vi, mask);
        Reg::Muls(index, output, uint32_t{2}, mask);
        Reg::Adds(index, index, 4U * prev, mask);
        Reg::Scatter(dst, cr, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(dst, ci, index, mask);
        Reg::Add(cr, ar, bi, mask);
        Reg::Sub(ci, ai, br, mask);
        Reg::Muls(index, output, uint32_t{2}, mask);
        Reg::Adds(index, index, 6U * prev, mask);
        Reg::Scatter(dst, cr, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(dst, ci, index, mask);
    }
}
} // namespace fft_mixed
