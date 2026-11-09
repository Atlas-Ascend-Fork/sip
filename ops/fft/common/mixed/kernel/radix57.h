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
// One lane owns a butterfly. Pair z[p] with z[R-p] after external twiddles.
// Scalar block/first loops also support prev with preceding odd factors.

__no_simd_vf_fusion__ __simd_vf__ inline void Radix5(__ubuf__ float* src, __ubuf__ float* dst, __ubuf__ float* roots,
                                                     uint32_t size, uint32_t rootN, uint32_t prev, bool inverse)
{
    using namespace AscendC;
    const uint32_t part = size / 5U, step = rootN / (5U * prev);
    const float sign = inverse ? 1.0f : -1.0f;
    for (uint32_t block = 0; block < size / (5U * prev); ++block) {
        for (uint32_t first = 0; first < prev; first += 64U) {
            uint32_t active = prev - first < 64U ? prev - first : 64U;
            Reg::MaskReg mask = Reg::UpdateMask<float>(active);
            Reg::RegTensor<uint32_t> j, index;
            Reg::RegTensor<float> zr, zi, ar, ai, br, bi, wr, wi, tmp, real;
            Reg::RegTensor<float> er, ei, orr, oi;
            Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(j), int32_t(first));
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * block * prev, mask);
            Reg::Gather(zr, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(zi, src, index, mask);

            Reg::RegTensor<float> s1r, s1i, d1r, d1i;
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * (block * prev + 1U * part), mask);
            Reg::Gather(ar, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(ai, src, index, mask);
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
            Reg::Gather(br, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(bi, src, index, mask);
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
            Reg::Gather(ar, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(ai, src, index, mask);
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
            Reg::Gather(br, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(bi, src, index, mask);
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
            Reg::Scatter(dst, er, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ei, index, mask);
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
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
            Reg::Add(ar, er, oi, mask);
            Reg::Sub(ai, ei, orr, mask);
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * (5U * block + 4U) * prev, mask);
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
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
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
            Reg::Add(ar, er, oi, mask);
            Reg::Sub(ai, ei, orr, mask);
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * (5U * block + 3U) * prev, mask);
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
        }
    }
}

__no_simd_vf_fusion__ __simd_vf__ inline void Radix7(__ubuf__ float* src, __ubuf__ float* dst, __ubuf__ float* roots,
                                                     uint32_t size, uint32_t rootN, uint32_t prev, bool inverse)
{
    using namespace AscendC;
    const uint32_t part = size / 7U, step = rootN / (7U * prev);
    const float sign = inverse ? 1.0f : -1.0f;
    for (uint32_t block = 0; block < size / (7U * prev); ++block) {
        for (uint32_t first = 0; first < prev; first += 64U) {
            uint32_t active = prev - first < 64U ? prev - first : 64U;
            Reg::MaskReg mask = Reg::UpdateMask<float>(active);
            Reg::RegTensor<uint32_t> j, index;
            Reg::RegTensor<float> zr, zi, ar, ai, br, bi, wr, wi, tmp, real;
            Reg::RegTensor<float> er, ei, orr, oi;
            Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(j), int32_t(first));
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * block * prev, mask);
            Reg::Gather(zr, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(zi, src, index, mask);

            Reg::RegTensor<float> s1r, s1i, d1r, d1i;
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * (block * prev + 1U * part), mask);
            Reg::Gather(ar, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(ai, src, index, mask);
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
            Reg::Gather(br, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(bi, src, index, mask);
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
            Reg::Gather(ar, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(ai, src, index, mask);
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
            Reg::Gather(br, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(bi, src, index, mask);
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
            Reg::Gather(ar, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(ai, src, index, mask);
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
            Reg::Gather(br, src, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(bi, src, index, mask);
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
            Reg::Scatter(dst, er, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ei, index, mask);
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
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
            Reg::Add(ar, er, oi, mask);
            Reg::Sub(ai, ei, orr, mask);
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * (7U * block + 6U) * prev, mask);
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
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
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
            Reg::Add(ar, er, oi, mask);
            Reg::Sub(ai, ei, orr, mask);
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * (7U * block + 5U) * prev, mask);
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
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
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
            Reg::Add(ar, er, oi, mask);
            Reg::Sub(ai, ei, orr, mask);
            Reg::Muls(index, j, uint32_t{2}, mask);
            Reg::Adds(index, index, 2U * (7U * block + 4U) * prev, mask);
            Reg::Scatter(dst, ar, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, ai, index, mask);
        }
    }
}

} // namespace fft_mixed
