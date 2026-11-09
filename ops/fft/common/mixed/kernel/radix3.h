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
// One SIMD lane owns all three outputs of a Stockham radix-3 butterfly.
// The outer block loop avoids vector integer division for repeated odd factors.
__no_simd_vf_fusion__ __simd_vf__ inline void Radix3(__ubuf__ float* src, __ubuf__ float* dst, __ubuf__ float* roots,
                                                     uint32_t size, uint32_t rootN, uint32_t prev, bool inverse)
{
    using namespace AscendC;
    const uint32_t part = size / 3U;
    const uint32_t step = rootN / (3U * prev);
    const float sine = inverse ? 0.8660254037844386f : -0.8660254037844386f;
    for (uint32_t first = 0; first < prev; first += 64U) {
        uint32_t active = prev - first < 64U ? prev - first : 64U;
        Reg::MaskReg mask = Reg::UpdateMask<float>(active);
        Reg::RegTensor<uint32_t> j, index, coeffIndex;
        Reg::RegTensor<float> ar, ai, br, bi, cr, ci, wr, wi, temp, real;
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(j), int32_t(first));
        Reg::RegTensor<float> w1r, w1i, w2r, w2i;
        Reg::Muls(coeffIndex, j, 2U * step, mask);
        Reg::Gather(w1r, roots, coeffIndex, mask);
        Reg::Adds(index, coeffIndex, uint32_t{1}, mask);
        Reg::Gather(w1i, roots, index, mask);
        Reg::Muls(coeffIndex, coeffIndex, uint32_t{2}, mask);
        Reg::Gather(w2r, roots, coeffIndex, mask);
        Reg::Adds(index, coeffIndex, uint32_t{1}, mask);
        Reg::Gather(w2i, roots, index, mask);
        for (uint32_t block = 0; block < size / (3U * prev); ++block) {
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * block * prev, mask);
            Reg::Gather(ar, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(ai, src, index, mask);
            Reg::Adds(index, index, 2U * part - 1U, mask);
            Reg::Gather(br, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(bi, src, index, mask);
            Reg::Adds(index, index, 2U * part - 1U, mask);
            Reg::Gather(cr, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(ci, src, index, mask);

            Reg::Adds(wr, w1r, 0.0f, mask);
            Reg::Adds(wi, w1i, 0.0f, mask);
            Reg::Mul(real, br, wr, mask);
            Reg::Mul(temp, bi, wi, mask);
            Reg::Sub(real, real, temp, mask);
            Reg::Mul(temp, br, wi, mask);
            Reg::Mul(bi, bi, wr, mask);
            Reg::Add(bi, bi, temp, mask);
            Reg::Adds(br, real, 0.0f, mask);
            Reg::Adds(wr, w2r, 0.0f, mask);
            Reg::Adds(wi, w2i, 0.0f, mask);
            Reg::Mul(real, cr, wr, mask);
            Reg::Mul(temp, ci, wi, mask);
            Reg::Sub(real, real, temp, mask);
            Reg::Mul(temp, cr, wi, mask);
            Reg::Mul(ci, ci, wr, mask);
            Reg::Add(ci, ci, temp, mask);
            Reg::Adds(cr, real, 0.0f, mask);

            // wr/wi = b+c; cr/ci = sin(2*pi/3)*(b-c).
            Reg::Add(wr, br, cr, mask);
            Reg::Add(wi, bi, ci, mask);
            Reg::Sub(cr, br, cr, mask);
            Reg::Sub(ci, bi, ci, mask);
            Reg::Muls(cr, cr, sine, mask);
            Reg::Muls(ci, ci, sine, mask);
            Reg::Add(br, ar, wr, mask);
            Reg::Add(bi, ai, wi, mask);
            Reg::Muls(wr, wr, -0.5f, mask);
            Reg::Muls(wi, wi, -0.5f, mask);
            Reg::Add(ar, ar, wr, mask);
            Reg::Add(ai, ai, wi, mask);
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 6U * block * prev, mask);
            Reg::Scatter(dst, br, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, bi, index, mask);
            Reg::Sub(br, ar, ci, mask);
            Reg::Add(bi, ai, cr, mask);
            Reg::Adds(index, index, 2U * prev - 1U, mask);
            Reg::Scatter(dst, br, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, bi, index, mask);
            Reg::Add(br, ar, ci, mask);
            Reg::Sub(bi, ai, cr, mask);
            Reg::Adds(index, index, 2U * prev - 1U, mask);
            Reg::Scatter(dst, br, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, bi, index, mask);
        }
    }
}
} // namespace fft_mixed
