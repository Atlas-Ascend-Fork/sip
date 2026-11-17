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
// Pack 1/2/4 rows into 64 SIMD lanes; preserve per-row butterfly coordinates.
__simd_callee__ inline void TileIndices(AscendC::Reg::RegTensor<uint32_t>& coordinate,
                                        AscendC::Reg::RegTensor<uint32_t>& rowOffset, AscendC::Reg::MaskReg& mask,
                                        uint32_t first, uint32_t limit, uint32_t rowBase, uint32_t rows,
                                        uint32_t rowStride, uint32_t cols)
{
    using namespace AscendC;
    const uint32_t packed = 64U / cols;
    uint32_t active = (rows - rowBase < packed ? rows - rowBase : packed) * cols;
    Reg::MaskReg rowMask = Reg::UpdateMask<float>(active);
    Reg::RegTensor<uint32_t> lane, bits;
    Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), 0);
    Reg::Duplicate(bits, cols - 1U, rowMask);
    Reg::And(coordinate, lane, bits, rowMask);
    Reg::Adds(coordinate, coordinate, first, rowMask);
    const int16_t shift = cols == 16 ? 4 : cols == 32 ? 5 : 6;
    Reg::ShiftRights<uint32_t, int16_t>(rowOffset, lane, shift, rowMask);
    Reg::Adds(rowOffset, rowOffset, rowBase, rowMask);
    Reg::Muls(rowOffset, rowOffset, rowStride, rowMask);
    Reg::Compares<uint32_t, CMPMODE::LT>(mask, coordinate, limit, rowMask);
}

__simd_callee__ inline void TileRadix2Body(__ubuf__ float* src, __ubuf__ float* dst, __ubuf__ float* roots,
                                           uint32_t size, uint32_t rootN, uint32_t half, uint32_t rows,
                                           uint32_t rowStride)
{
    using namespace AscendC;
    const uint32_t cols = rows >= 4 ? 16U : rows >= 2 ? 32U : 64U;
    const uint32_t part = size / 2;
    for (uint32_t rowBase = 0; rowBase < rows; rowBase += 64U / cols) {
        for (uint32_t first = 0; first < part; first += cols) {
            Reg::MaskReg mask;
            Reg::RegTensor<uint32_t> q, j, halfMask, index, output0;
            Reg::RegTensor<float> ur, ui, vr, vi, wr, wi, tr, ti, temporary;
            Reg::RegTensor<uint32_t> rowOffset, memoryIndex;
            TileIndices(q, rowOffset, mask, first, part, rowBase, rows, rowStride, cols);
            Reg::Duplicate(halfMask, half - 1U, mask);
            Reg::And(j, q, halfMask, mask);
            Reg::Muls(index, q, uint32_t{2}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(ur, src, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(ui, src, memoryIndex, mask);
            Reg::Adds(index, index, size - 1U, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(vr, src, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(vi, src, memoryIndex, mask);
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
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, ur, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, ui, memoryIndex, mask);
            Reg::Adds(index, index, 2U * half - 1U, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, vr, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, vi, memoryIndex, mask);
        }
    }
}
__simd_callee__ inline void TileRadix4(__ubuf__ float* src, __ubuf__ float* dst, __ubuf__ float* roots, uint32_t size,
                                       uint32_t rootN, uint32_t prev, bool inverse, uint32_t rows, uint32_t rowStride)
{
    using namespace AscendC;
    const uint32_t cols = rows >= 4 ? 16U : rows >= 2 ? 32U : 64U;
    const uint32_t part = size / 4U, step = rootN / (4U * prev);
    const float sign = inverse ? 1.0f : -1.0f;
    for (uint32_t rowBase = 0; rowBase < rows; rowBase += 64U / cols) {
        for (uint32_t first = 0; first < part; first += cols) {
            Reg::MaskReg mask;
            Reg::RegTensor<uint32_t> q, j, bits, index, output;
            Reg::RegTensor<float> ar, ai, br, bi, cr, ci, dr, di;
            Reg::RegTensor<float> wr, wi, real, temp, ur, ui, vr, vi;
            Reg::RegTensor<uint32_t> rowOffset, memoryIndex;
            TileIndices(q, rowOffset, mask, first, part, rowBase, rows, rowStride, cols);
            Reg::Duplicate(bits, prev - 1U, mask);
            Reg::And(j, q, bits, mask);
            Reg::Muls(index, q, uint32_t{2}, mask);
            Reg::Adds(index, index, 0U * part, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(ar, src, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(ai, src, memoryIndex, mask);
            Reg::Muls(index, q, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * part, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(br, src, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(bi, src, memoryIndex, mask);
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
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(cr, src, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(ci, src, memoryIndex, mask);
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
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(dr, src, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Gather(di, src, memoryIndex, mask);
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
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, cr, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, ci, memoryIndex, mask);
            Reg::Sub(cr, ar, bi, mask);
            Reg::Add(ci, ai, br, mask);
            Reg::Muls(index, output, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * prev, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, cr, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, ci, memoryIndex, mask);
            Reg::Sub(cr, ur, vr, mask);
            Reg::Sub(ci, ui, vi, mask);
            Reg::Muls(index, output, uint32_t{2}, mask);
            Reg::Adds(index, index, 4U * prev, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, cr, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, ci, memoryIndex, mask);
            Reg::Add(cr, ar, bi, mask);
            Reg::Sub(ci, ai, br, mask);
            Reg::Muls(index, output, uint32_t{2}, mask);
            Reg::Adds(index, index, 6U * prev, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, cr, memoryIndex, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Add(memoryIndex, index, rowOffset, mask);
            Reg::Scatter(dst, ci, memoryIndex, mask);
        }
    }
}
__no_simd_vf_fusion__ __simd_vf__ inline void TileRadix3(__ubuf__ float* src, __ubuf__ float* dst,
                                                         __ubuf__ float* roots, uint32_t size, uint32_t rootN,
                                                         uint32_t prev, bool inverse, uint32_t rows, uint32_t rowStride)
{
    using namespace AscendC;
    const uint32_t cols = rows >= 4 ? 16U : rows >= 2 ? 32U : 64U;
    const uint32_t part = size / 3U;
    const uint32_t step = rootN / (3U * prev);
    const float sine = inverse ? 0.8660254037844386f : -0.8660254037844386f;
    for (uint32_t rowBase = 0; rowBase < rows; rowBase += 64U / cols) {
        for (uint32_t first = 0; first < prev; first += cols) {
            Reg::MaskReg mask;
            Reg::RegTensor<uint32_t> j, index, coeffIndex;
            Reg::RegTensor<float> ar, ai, br, bi, cr, ci, wr, wi, temp, real;
            Reg::RegTensor<uint32_t> rowOffset, memoryIndex;
            TileIndices(j, rowOffset, mask, first, prev, rowBase, rows, rowStride, cols);
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
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ar, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ai, src, memoryIndex, mask);
                Reg::Adds(index, index, 2U * part - 1U, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(br, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(bi, src, memoryIndex, mask);
                Reg::Adds(index, index, 2U * part - 1U, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(cr, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ci, src, memoryIndex, mask);

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
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, br, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, bi, memoryIndex, mask);
                Reg::Sub(br, ar, ci, mask);
                Reg::Add(bi, ai, cr, mask);
                Reg::Adds(index, index, 2U * prev - 1U, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, br, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, bi, memoryIndex, mask);
                Reg::Add(br, ar, ci, mask);
                Reg::Sub(bi, ai, cr, mask);
                Reg::Adds(index, index, 2U * prev - 1U, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, br, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, bi, memoryIndex, mask);
            }
        }
    }
}
__no_simd_vf_fusion__ __simd_vf__ inline void TileRadix5(__ubuf__ float* src, __ubuf__ float* dst,
                                                         __ubuf__ float* roots, uint32_t size, uint32_t rootN,
                                                         uint32_t prev, bool inverse, uint32_t rows, uint32_t rowStride)
{
    using namespace AscendC;
    const uint32_t cols = rows >= 4 ? 16U : rows >= 2 ? 32U : 64U;
    const uint32_t part = size / 5U, step = rootN / (5U * prev);
    const float sign = inverse ? 1.0f : -1.0f;
    for (uint32_t rowBase = 0; rowBase < rows; rowBase += 64U / cols) {
        for (uint32_t block = 0; block < size / (5U * prev); ++block) {
            for (uint32_t first = 0; first < prev; first += cols) {
                Reg::MaskReg mask;
                Reg::RegTensor<uint32_t> j, index;
                Reg::RegTensor<float> zr, zi, ar, ai, br, bi, wr, wi, tmp, real;
                Reg::RegTensor<float> er, ei, orr, oi;
                Reg::RegTensor<uint32_t> rowOffset, memoryIndex;
                TileIndices(j, rowOffset, mask, first, prev, rowBase, rows, rowStride, cols);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * block * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(zr, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(zi, src, memoryIndex, mask);

                Reg::RegTensor<float> s1r, s1i, d1r, d1i;
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 1U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ar, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ai, src, memoryIndex, mask);
                Reg::Muls(index, j, 2U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, ar, wr, mask);
                Reg::Mul(tmp, ai, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, ar, wi, mask);
                Reg::Mul(ai, ai, wr, mask);
                Reg::Add(ai, ai, tmp, mask);
                Reg::Adds(ar, real, 0.0f, mask);

                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 4U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(br, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(bi, src, memoryIndex, mask);
                Reg::Muls(index, j, 8U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, br, wr, mask);
                Reg::Mul(tmp, bi, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, br, wi, mask);
                Reg::Mul(bi, bi, wr, mask);
                Reg::Add(bi, bi, tmp, mask);
                Reg::Adds(br, real, 0.0f, mask);

                Reg::Add(s1r, ar, br, mask);
                Reg::Add(s1i, ai, bi, mask);
                Reg::Sub(d1r, ar, br, mask);
                Reg::Sub(d1i, ai, bi, mask);
                Reg::RegTensor<float> s2r, s2i, d2r, d2i;
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 2U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ar, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ai, src, memoryIndex, mask);
                Reg::Muls(index, j, 4U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, ar, wr, mask);
                Reg::Mul(tmp, ai, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, ar, wi, mask);
                Reg::Mul(ai, ai, wr, mask);
                Reg::Add(ai, ai, tmp, mask);
                Reg::Adds(ar, real, 0.0f, mask);

                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 3U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(br, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(bi, src, memoryIndex, mask);
                Reg::Muls(index, j, 6U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, br, wr, mask);
                Reg::Mul(tmp, bi, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, br, wi, mask);
                Reg::Mul(bi, bi, wr, mask);
                Reg::Add(bi, bi, tmp, mask);
                Reg::Adds(br, real, 0.0f, mask);

                Reg::Add(s2r, ar, br, mask);
                Reg::Add(s2i, ai, bi, mask);
                Reg::Sub(d2r, ar, br, mask);
                Reg::Sub(d2i, ai, bi, mask);
                Reg::Adds(er, zr, 0.0f, mask);
                Reg::Adds(ei, zi, 0.0f, mask);
                Reg::Add(er, er, s1r, mask);
                Reg::Add(ei, ei, s1i, mask);
                Reg::Add(er, er, s2r, mask);
                Reg::Add(ei, ei, s2i, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (5U * block + 0U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, er, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ei, memoryIndex, mask);
                Reg::Adds(er, zr, 0.0f, mask);
                Reg::Adds(ei, zi, 0.0f, mask);
                Reg::Muls(tmp, s1r, 0.30901699437494745f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s1i, 0.30901699437494745f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(orr, d1r, sign * 0.95105651629515353f, mask);
                Reg::Muls(oi, d1i, sign * 0.95105651629515353f, mask);
                Reg::Muls(tmp, s2r, -0.80901699437494734f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s2i, -0.80901699437494734f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(tmp, d2r, sign * 0.58778525229247325f, mask);
                Reg::Add(orr, orr, tmp, mask);
                Reg::Muls(tmp, d2i, sign * 0.58778525229247325f, mask);
                Reg::Add(oi, oi, tmp, mask);
                Reg::Sub(ar, er, oi, mask);
                Reg::Add(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (5U * block + 1U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
                Reg::Add(ar, er, oi, mask);
                Reg::Sub(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (5U * block + 4U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
                Reg::Adds(er, zr, 0.0f, mask);
                Reg::Adds(ei, zi, 0.0f, mask);
                Reg::Muls(tmp, s1r, -0.80901699437494734f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s1i, -0.80901699437494734f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(orr, d1r, sign * 0.58778525229247325f, mask);
                Reg::Muls(oi, d1i, sign * 0.58778525229247325f, mask);
                Reg::Muls(tmp, s2r, 0.30901699437494723f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s2i, 0.30901699437494723f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(tmp, d2r, sign * -0.95105651629515364f, mask);
                Reg::Add(orr, orr, tmp, mask);
                Reg::Muls(tmp, d2i, sign * -0.95105651629515364f, mask);
                Reg::Add(oi, oi, tmp, mask);
                Reg::Sub(ar, er, oi, mask);
                Reg::Add(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (5U * block + 2U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
                Reg::Add(ar, er, oi, mask);
                Reg::Sub(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (5U * block + 3U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
            }
        }
    }
}
__no_simd_vf_fusion__ __simd_vf__ inline void TileRadix7(__ubuf__ float* src, __ubuf__ float* dst,
                                                         __ubuf__ float* roots, uint32_t size, uint32_t rootN,
                                                         uint32_t prev, bool inverse, uint32_t rows, uint32_t rowStride)
{
    using namespace AscendC;
    const uint32_t cols = rows >= 4 ? 16U : rows >= 2 ? 32U : 64U;
    const uint32_t part = size / 7U, step = rootN / (7U * prev);
    const float sign = inverse ? 1.0f : -1.0f;
    for (uint32_t rowBase = 0; rowBase < rows; rowBase += 64U / cols) {
        for (uint32_t block = 0; block < size / (7U * prev); ++block) {
            for (uint32_t first = 0; first < prev; first += cols) {
                Reg::MaskReg mask;
                Reg::RegTensor<uint32_t> j, index;
                Reg::RegTensor<float> zr, zi, ar, ai, br, bi, wr, wi, tmp, real;
                Reg::RegTensor<float> er, ei, orr, oi;
                Reg::RegTensor<uint32_t> rowOffset, memoryIndex;
                TileIndices(j, rowOffset, mask, first, prev, rowBase, rows, rowStride, cols);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * block * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(zr, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(zi, src, memoryIndex, mask);

                Reg::RegTensor<float> s1r, s1i, d1r, d1i;
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 1U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ar, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ai, src, memoryIndex, mask);
                Reg::Muls(index, j, 2U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, ar, wr, mask);
                Reg::Mul(tmp, ai, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, ar, wi, mask);
                Reg::Mul(ai, ai, wr, mask);
                Reg::Add(ai, ai, tmp, mask);
                Reg::Adds(ar, real, 0.0f, mask);

                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 6U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(br, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(bi, src, memoryIndex, mask);
                Reg::Muls(index, j, 12U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, br, wr, mask);
                Reg::Mul(tmp, bi, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, br, wi, mask);
                Reg::Mul(bi, bi, wr, mask);
                Reg::Add(bi, bi, tmp, mask);
                Reg::Adds(br, real, 0.0f, mask);

                Reg::Add(s1r, ar, br, mask);
                Reg::Add(s1i, ai, bi, mask);
                Reg::Sub(d1r, ar, br, mask);
                Reg::Sub(d1i, ai, bi, mask);
                Reg::RegTensor<float> s2r, s2i, d2r, d2i;
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 2U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ar, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ai, src, memoryIndex, mask);
                Reg::Muls(index, j, 4U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, ar, wr, mask);
                Reg::Mul(tmp, ai, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, ar, wi, mask);
                Reg::Mul(ai, ai, wr, mask);
                Reg::Add(ai, ai, tmp, mask);
                Reg::Adds(ar, real, 0.0f, mask);

                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 5U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(br, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(bi, src, memoryIndex, mask);
                Reg::Muls(index, j, 10U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, br, wr, mask);
                Reg::Mul(tmp, bi, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, br, wi, mask);
                Reg::Mul(bi, bi, wr, mask);
                Reg::Add(bi, bi, tmp, mask);
                Reg::Adds(br, real, 0.0f, mask);

                Reg::Add(s2r, ar, br, mask);
                Reg::Add(s2i, ai, bi, mask);
                Reg::Sub(d2r, ar, br, mask);
                Reg::Sub(d2i, ai, bi, mask);
                Reg::RegTensor<float> s3r, s3i, d3r, d3i;
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 3U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ar, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(ai, src, memoryIndex, mask);
                Reg::Muls(index, j, 6U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, ar, wr, mask);
                Reg::Mul(tmp, ai, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, ar, wi, mask);
                Reg::Mul(ai, ai, wr, mask);
                Reg::Add(ai, ai, tmp, mask);
                Reg::Adds(ar, real, 0.0f, mask);

                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (block * prev + 4U * part), mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(br, src, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Gather(bi, src, memoryIndex, mask);
                Reg::Muls(index, j, 8U * step, mask);
                Reg::Gather(wr, roots, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, roots, index, mask);
                Reg::Mul(real, br, wr, mask);
                Reg::Mul(tmp, bi, wi, mask);
                Reg::Sub(real, real, tmp, mask);
                Reg::Mul(tmp, br, wi, mask);
                Reg::Mul(bi, bi, wr, mask);
                Reg::Add(bi, bi, tmp, mask);
                Reg::Adds(br, real, 0.0f, mask);

                Reg::Add(s3r, ar, br, mask);
                Reg::Add(s3i, ai, bi, mask);
                Reg::Sub(d3r, ar, br, mask);
                Reg::Sub(d3i, ai, bi, mask);
                Reg::Adds(er, zr, 0.0f, mask);
                Reg::Adds(ei, zi, 0.0f, mask);
                Reg::Add(er, er, s1r, mask);
                Reg::Add(ei, ei, s1i, mask);
                Reg::Add(er, er, s2r, mask);
                Reg::Add(ei, ei, s2i, mask);
                Reg::Add(er, er, s3r, mask);
                Reg::Add(ei, ei, s3i, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (7U * block + 0U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, er, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ei, memoryIndex, mask);
                Reg::Adds(er, zr, 0.0f, mask);
                Reg::Adds(ei, zi, 0.0f, mask);
                Reg::Muls(tmp, s1r, 0.62348980185873359f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s1i, 0.62348980185873359f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(orr, d1r, sign * 0.7818314824680298f, mask);
                Reg::Muls(oi, d1i, sign * 0.7818314824680298f, mask);
                Reg::Muls(tmp, s2r, -0.22252093395631434f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s2i, -0.22252093395631434f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(tmp, d2r, sign * 0.97492791218182362f, mask);
                Reg::Add(orr, orr, tmp, mask);
                Reg::Muls(tmp, d2i, sign * 0.97492791218182362f, mask);
                Reg::Add(oi, oi, tmp, mask);
                Reg::Muls(tmp, s3r, -0.90096886790241903f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s3i, -0.90096886790241903f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(tmp, d3r, sign * 0.43388373911755823f, mask);
                Reg::Add(orr, orr, tmp, mask);
                Reg::Muls(tmp, d3i, sign * 0.43388373911755823f, mask);
                Reg::Add(oi, oi, tmp, mask);
                Reg::Sub(ar, er, oi, mask);
                Reg::Add(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (7U * block + 1U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
                Reg::Add(ar, er, oi, mask);
                Reg::Sub(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (7U * block + 6U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
                Reg::Adds(er, zr, 0.0f, mask);
                Reg::Adds(ei, zi, 0.0f, mask);
                Reg::Muls(tmp, s1r, -0.22252093395631434f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s1i, -0.22252093395631434f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(orr, d1r, sign * 0.97492791218182362f, mask);
                Reg::Muls(oi, d1i, sign * 0.97492791218182362f, mask);
                Reg::Muls(tmp, s2r, -0.90096886790241915f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s2i, -0.90096886790241915f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(tmp, d2r, sign * -0.43388373911755801f, mask);
                Reg::Add(orr, orr, tmp, mask);
                Reg::Muls(tmp, d2i, sign * -0.43388373911755801f, mask);
                Reg::Add(oi, oi, tmp, mask);
                Reg::Muls(tmp, s3r, 0.62348980185873337f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s3i, 0.62348980185873337f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(tmp, d3r, sign * -0.78183148246802991f, mask);
                Reg::Add(orr, orr, tmp, mask);
                Reg::Muls(tmp, d3i, sign * -0.78183148246802991f, mask);
                Reg::Add(oi, oi, tmp, mask);
                Reg::Sub(ar, er, oi, mask);
                Reg::Add(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (7U * block + 2U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
                Reg::Add(ar, er, oi, mask);
                Reg::Sub(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (7U * block + 5U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
                Reg::Adds(er, zr, 0.0f, mask);
                Reg::Adds(ei, zi, 0.0f, mask);
                Reg::Muls(tmp, s1r, -0.90096886790241903f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s1i, -0.90096886790241903f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(orr, d1r, sign * 0.43388373911755823f, mask);
                Reg::Muls(oi, d1i, sign * 0.43388373911755823f, mask);
                Reg::Muls(tmp, s2r, 0.62348980185873337f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s2i, 0.62348980185873337f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(tmp, d2r, sign * -0.78183148246802991f, mask);
                Reg::Add(orr, orr, tmp, mask);
                Reg::Muls(tmp, d2i, sign * -0.78183148246802991f, mask);
                Reg::Add(oi, oi, tmp, mask);
                Reg::Muls(tmp, s3r, -0.22252093395631409f, mask);
                Reg::Add(er, er, tmp, mask);
                Reg::Muls(tmp, s3i, -0.22252093395631409f, mask);
                Reg::Add(ei, ei, tmp, mask);
                Reg::Muls(tmp, d3r, sign * 0.97492791218182373f, mask);
                Reg::Add(orr, orr, tmp, mask);
                Reg::Muls(tmp, d3i, sign * 0.97492791218182373f, mask);
                Reg::Add(oi, oi, tmp, mask);
                Reg::Sub(ar, er, oi, mask);
                Reg::Add(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (7U * block + 3U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
                Reg::Add(ar, er, oi, mask);
                Reg::Sub(ai, ei, orr, mask);
                Reg::Muls(index, j, uint32_t{2}, mask);
                Reg::Adds(index, index, 2U * (7U * block + 4U) * prev, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ar, memoryIndex, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Add(memoryIndex, index, rowOffset, mask);
                Reg::Scatter(dst, ai, memoryIndex, mask);
            }
        }
    }
}
__no_simd_vf_fusion__ __simd_vf__ inline void TilePrefix(__ubuf__ float* src, __ubuf__ float* dst,
                                                         __ubuf__ float* roots, uint32_t size, uint32_t rootN,
                                                         int stages, bool inverse, uint32_t rows, uint32_t rowStride)
{
    uint32_t half = 1;
    for (int stage = 0; stage < stages; ++stage) {
        if (stage + 1 < stages) {
            TileRadix4(src, dst, roots, size, rootN, half, inverse, rows, rowStride);
            ++stage;
            half *= 2;
        } else {
            TileRadix2Body(src, dst, roots, size, rootN, half, rows, rowStride);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        auto* tmp = src;
        src = dst;
        dst = tmp;
        half *= 2;
    }
}
} // namespace fft_mixed
