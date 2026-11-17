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
#include <cstdint>
#include <string>
namespace AsdSip::OpParam {
enum class MixedCommonPass : int32_t { Complex, Real, Hermitian };
struct FftMixedCommon {
    int64_t n = 0, rows = 0;
    MixedCommonPass pass = MixedCommonPass::Complex;
    bool inverse = false;
    bool operator==(const FftMixedCommon& p) const
    {
        return n == p.n && rows == p.rows && pass == p.pass && inverse == p.inverse;
    }
    std::string ToString() const
    {
        return "FftMixedCommon n=" + std::to_string(n) + " rows=" + std::to_string(rows) +
               " pass=" + std::to_string(int(pass));
    }
};
inline bool MixedCommonLength(int64_t n) { return n >= 32 && n <= 8192 && n % 32 == 0 && (n & (n - 1)) != 0; }
inline bool ValidMixedCommon(const FftMixedCommon& p)
{
    return MixedCommonLength(p.n) && p.rows >= 1 && p.rows <= 8192 && int(p.pass) >= 0 && int(p.pass) <= 2 &&
           (p.pass != MixedCommonPass::Real || !p.inverse) && (p.pass != MixedCommonPass::Hermitian || p.inverse);
}
// Keep every eligible mixed pass on the capacity-based scheduler, including
// one-row tiles. Two data regions remain when three no longer fit a full row.
struct MixedCommonTilePlan {
    int key, buffers, capacity, rows;
};
inline MixedCommonTilePlan SelectMixedCommonTile(const FftMixedCommon& p)
{
    if (!ValidMixedCommon(p))
        return {-1, 0, 0, 0};
    constexpr int arenaFloats = 212 * 1024 / sizeof(float);
    const bool real = p.pass != MixedCommonPass::Complex;
    const int stride = (real ? p.n : 2 * p.n) + 128;
    int buffers;
    if (real)
        buffers = p.n > 5312 && p.n <= 6688 ? 3 : 4;
    else if (p.n <= 2944)
        buffers = 4;
    else
        buffers = ((arenaFloats - 2 * p.n) / 3 / 8) * 8 >= stride ? 3 : 2;
    const int capacity = ((arenaFloats - 2 * p.n) / buffers / 8) * 8;
    const int rows = capacity / stride;
    if (rows == 0)
        return {-1, 0, 0, 0};
    const int key = real ? (buffers == 3 ? 25 : 21) : buffers == 4 ? 19 : buffers == 3 ? 22 : 26;
    return {key, buffers, capacity, rows};
}
inline int64_t MixedCommonFloats(const FftMixedCommon& p, bool output)
{
    if (p.pass == MixedCommonPass::Real)
        return p.rows * (output ? p.n + 2 : p.n);
    if (p.pass == MixedCommonPass::Hermitian)
        return p.rows * (output ? p.n : p.n + 2);
    return 2 * p.rows * p.n;
}
inline int64_t MixedCommonCoeffFloats(const FftMixedCommon& p) { return 2 * p.n; }
struct MixedCommonTiling {
    int32_t n, rows, pass, inverse, radix2Stages, oddStages, twoFrequencies;
    int32_t radix[8], prev[8];
};
inline bool BuildMixedCommonTiling(const FftMixedCommon& p, MixedCommonTiling& t)
{
    t = {};
    if (!ValidMixedCommon(p))
        return false;
    t.n = int32_t(p.n);
    t.rows = int32_t(p.rows);
    t.pass = int32_t(p.pass);
    t.inverse = p.inverse;
    // Use paired symmetric outputs for larger transforms.
    t.twoFrequencies = p.n >= 1824;
    int left = p.pass == MixedCommonPass::Complex ? t.n : t.n / 2, prev = 1;
    while (left % 2 == 0) {
        ++t.radix2Stages;
        left /= 2;
        prev *= 2;
    }
    for (int r = 3; left > 1; r += 2) {
        while (left % r == 0) {
            if (t.oddStages == 8 || r > 251)
                return false;
            t.radix[t.oddStages] = r;
            t.prev[t.oddStages++] = prev;
            left /= r;
            prev *= r;
        }
    }
    return t.oddStages > 0;
}
} // namespace AsdSip::OpParam
