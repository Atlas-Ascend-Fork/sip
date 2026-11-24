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
#include <initializer_list>
#include <string>
#include "fft_real_target_routes.h"
namespace AsdSip::OpParam {
enum class MixedCommonPass : int32_t { Complex, Real, Hermitian };
struct FftMixedCommon {
    int64_t n = 0, rows = 0;
    MixedCommonPass pass = MixedCommonPass::Complex;
    bool inverse = false;
    bool complex2dAxis = false;
    bool cubeTailAllowed = false;
    bool allowRealCubePipeline = true;
    bool operator==(const FftMixedCommon& p) const
    {
        return n == p.n && rows == p.rows && pass == p.pass && inverse == p.inverse &&
               complex2dAxis == p.complex2dAxis && cubeTailAllowed == p.cubeTailAllowed &&
               allowRealCubePipeline == p.allowRealCubePipeline;
    }
    std::string ToString() const
    {
        return "FftMixedCommon n=" + std::to_string(n) + " rows=" + std::to_string(rows) +
               " pass=" + std::to_string(int(pass));
    }
};
inline bool MixedCommonLength(int64_t n) { return n >= 32 && n <= 8192 && n % 32 == 0 && (n & (n - 1)) != 0; }
inline int LargestPrimeFactor(int value)
{
    int largest = 1;
    for (int d = 2; d * d <= value; ++d) {
        while (value % d == 0) {
            largest = d;
            value /= d;
        }
    }
    return value > 1 ? value : largest;
}
inline bool RealMixedLength(int64_t n)
{
    // The measured mixed grid starts at 96. Shorter real transforms retain native dispatch.
    if (n < 96 || n > 8192 || n % 2 || (n & (n - 1)) == 0)
        return false;
    return LargestPrimeFactor(int(n / 2)) <= 251;
}
inline int FixedRealTargetTail(const FftMixedCommon& p)
{
    if (!p.cubeTailAllowed || p.rows != p.n || !MixedCommonLength(p.n))
        return -1;
    if (p.pass == MixedCommonPass::Hermitian && p.inverse) {
        if (p.n == 1632)
            return 51;
        if (p.n == 4000)
            return 125;
    } else if (p.pass != MixedCommonPass::Real || p.inverse) {
        return -1;
    }
    return REAL_TARGET_TAILS[p.n / 32 - 1];
}
struct CubeTailPlan {
    int tail = 0, lanes = 0, width = 0, kBlocks = 0, outBlocks = 0;
    bool direct = false;
    int rowsPerAivTile = 1;
    int TilingKey() const { return direct ? width : 1000 + 100 * (lanes / 8) + width; }
};
// n is a validated axis length in [1, 8192].
inline int SelectOddTailFactor(int n)
{
    int odd = n;
    while (odd % 2 == 0)
        odd /= 2;
    if (odd == 1)
        return 0;
    int factors[9] = {}, count = 0; // 3^9 exceeds the supported axis length.
    for (int d = 3; d * d <= odd; d += 2) {
        while (odd % d == 0) {
            factors[count++] = d;
            odd /= d;
        }
    }
    if (odd > 1)
        factors[count++] = odd;
    const int largest = factors[--count];
    if (largest > 129)
        return largest < 257 ? largest : 0;
    int tail = largest;
    // Once the smallest factor is <= 7, a composite needs a largest prime >= 17.
    while (count > 0) {
        const int d = factors[--count];
        if (largest < 17 && d <= 7)
            break;
        if (tail * d < 129)
            tail *= d;
    }
    return tail == largest && largest <= 23 ? 0 : tail;
}
// For each target axis N = 32 * k, store its measured C2C tail (zero selects Vec).
// The caller validates 1 <= n <= 8192. Both directions share the target table.
// Other supported axis lengths keep using the factor-based planner.
inline int CubeTailFactorForLength(int n)
{
    static constexpr uint8_t targetTails[] = {
        0,   0,   0,   0,  0,   0,   0,   0,  0,   0,   0,   0,  0,   0,   0,   0, // k = 1..16
        0,   0,   0,   0,  0,   0,   23,  0,  0,   0,   0,   0,  29,  0,   31,  0, // k = 17..32
        33,  0,   0,   0,  37,  0,   39,  0,  41,  0,   43,  0,  0,   0,   47,  0, // k = 33..48
        0,   0,   51,  0,  53,  0,   55,  0,  57,  0,   59,  0,  61,  31,  0,   0, // k = 49..64
        65,  0,   67,  0,  69,  0,   71,  0,  73,  0,   0,   0,  77,  0,   79,  0, // k = 65..80
        0,   0,   83,  0,  85,  43,  87,  0,  89,  0,   91,  0,  93,  47,  95,  0, // k = 81..96
        97,  0,   99,  0,  101, 0,   103, 0,  0,   53,  107, 0,  109, 0,   111, 0, // k = 97..112
        113, 0,   115, 0,  117, 59,  119, 0,  121, 61,  123, 31, 125, 0,   127, 0, // k = 113..128
        43,  0,   131, 0,  0,   67,  0,   0,  137, 0,   139, 0,  47,  71,  0,   0, // k = 129..144
        29,  73,  0,   0,  149, 0,   151, 0,  0,   0,   31,  0,  157, 79,  53,  0, // k = 145..160
        0,   0,   163, 41, 0,   83,  167, 0,  169, 85,  57,  43, 173, 87,  0,   0, // k = 161..176
        59,  89,  179, 0,  181, 0,   61,  0,  185, 93,  187, 47, 0,   95,  191, 0, // k = 177..192
        193, 97,  0,   0,  197, 0,   199, 0,  67,  101, 29,  0,  41,  103, 0,   0, // k = 193..208
        209, 0,   211, 53, 71,  107, 43,  0,  31,  109, 73,  0,  0,   0,   223, 0, // k = 209..224
        0,   113, 227, 0,  229, 0,   0,   0,  233, 0,   47,  59, 79,  0,   239, 0, // k = 225..240
        241, 121, 0,   61, 0,   123, 0,   31, 83,  0,   251, 0,  0,   127, 0,   0, // k = 241..256
    };
    static_assert(sizeof(targetTails) / sizeof(targetTails[0]) == 256);
    return n % 32 == 0 ? targetTails[n / 32 - 1] : SelectOddTailFactor(n);
}
inline int RealCubeUbBytes(int n, int rows, int lanes, int width)
{
    const int rootFloats = (2 * n + 511) / 512 * 512;
    const int rowFloats = (n + 128 + 511) / 512 * 512;
    return 4 * (rootFloats + 2 * rows * rowFloats) + 48 * lanes * width;
}
inline bool UseRealCubePipeline(const FftMixedCommon& p, const CubeTailPlan& cube)
{
    if (!p.allowRealCubePipeline || p.pass == MixedCommonPass::Complex || !cube.tail || cube.kBlocks != 1 ||
        cube.outBlocks != 1)
        return false;
    const int prefix = int(p.n / 2) / cube.tail;
    // A complete tile is sent to the AIC before its two FFT row buffers are reused.
    // Retain one complex prefix value per lane for each of the two in-flight tiles.
    return cube.rowsPerAivTile * prefix <= cube.lanes &&
           RealCubeUbBytes(int(p.n), cube.rowsPerAivTile, cube.lanes, cube.width) + 16 * cube.lanes <= 212 * 1024;
}
inline CubeTailPlan PlanRealCubeLayout(const FftMixedCommon& p, int tail)
{
    const int n = static_cast<int>(p.n), size = n / 2;
    if (tail < 3 || tail >= 257 || tail % 2 == 0 || size % tail != 0)
        return {};
    const int prefix = size / tail, half = tail / 2, width = (half / 16 + 1) * 16;
    const bool blocked = tail >= 129;
    CubeTailPlan best;
    int bestWork = 0, bestCalls = 0;
    for (int rows : {1, 2}) {
        // The first multirow form combines two 16-lane prefixes into one full tile.
        if (rows == 2 && (prefix != 16 || p.rows < 8))
            continue;
        for (int j : {32, 16, 8}) {
            if ((rows == 2 && j != 32) || (blocked && (rows * prefix) % j != 0))
                continue;
            for (int tile : {64, 48, 32, 16}) {
                if ((blocked && tile == 16) || (!blocked && tile > width))
                    continue;
                const int kb = (half + tile - 1) / tile, nb = (half + tile) / tile;
                // One complete N-root table, two compact data regions per row, and six
                // half-matrix fragments. Both AIVs share the two-slot L1 input region.
                if (RealCubeUbBytes(n, rows, j, tile) > 212 * 1024 || 64 * j * kb * tile > 128 * 1024)
                    continue;
                const int groups = (rows * prefix + j - 1) / j;
                const int work = groups * j * kb * nb * tile * tile, calls = groups * kb * nb;
                // Compare work and handshakes per real row, not per combined tile.
                const int scaledWork = work * best.rowsPerAivTile, scaledBestWork = bestWork * rows;
                const int scaledCalls = calls * best.rowsPerAivTile, scaledBestCalls = bestCalls * rows;
                // Small tails retain one complete matrix when it fits: splitting a padded
                // edge saves arithmetic but adds weight transfers and cross-core handshakes.
                const bool whole = !blocked && kb == 1 && nb == 1;
                const bool bestWhole = !blocked && best.kBlocks == 1 && best.outBlocks == 1;
                if (!best.tail || (whole && !bestWhole) ||
                    (whole == bestWhole && (scaledWork < scaledBestWork ||
                                            (scaledWork == scaledBestWork &&
                                             (scaledCalls < scaledBestCalls ||
                                              (scaledCalls == scaledBestCalls && rows > best.rowsPerAivTile)))))) {
                    best = {tail, j, tile, kb, nb, false, rows};
                    bestWork = work;
                    bestCalls = calls;
                }
            }
        }
    }
    return best;
}
inline bool PreferRealCubeFactor(int tail, int prefix, int largest)
{
    if (tail >= 129)
        return tail == largest || (prefix <= 32 && largest >= 43);
    if (tail == largest)
        return tail >= (prefix <= 16 ? 29 : 47);
    return prefix <= 32 ? tail >= 55 && largest >= 11 : tail >= 85 && largest >= 17;
}
inline CubeTailPlan SelectRealCubeCandidate(const FftMixedCommon& p, int fixed = -1)
{
    if (!p.cubeTailAllowed || !RealMixedLength(p.n) || p.rows < 1 || p.rows > 8192 ||
        (p.pass != MixedCommonPass::Real && p.pass != MixedCommonPass::Hermitian) ||
        (p.pass == MixedCommonPass::Real ? p.inverse : !p.inverse))
        return {};
    if (fixed >= 0)
        return PlanRealCubeLayout(p, fixed);

    // Keep the hardest odd radix in the Cube tail. Try single-matrix sizes before blocked sizes.
    const int size = int(p.n / 2), largest = LargestPrimeFactor(size);
    for (int limit : {127, 255}) {
        const int lower = limit == 127 ? largest : 129;
        for (int tail = limit; tail >= lower; tail -= 2) {
            if (tail % largest || size % tail || !PreferRealCubeFactor(tail, size / tail, largest))
                continue;
            const auto layout = PlanRealCubeLayout(p, tail);
            if (layout.tail)
                return layout;
        }
    }
    return {};
}
inline CubeTailPlan SelectRealCubeTail(const FftMixedCommon& p)
{
    return SelectRealCubeCandidate(p, FixedRealTargetTail(p));
}
inline CubeTailPlan SelectCubeTail(const FftMixedCommon& p)
{
    if (p.pass != MixedCommonPass::Complex)
        return SelectRealCubeTail(p);
    if (!p.cubeTailAllowed || p.pass != MixedCommonPass::Complex || p.n < 1 || p.n > 8192)
        return {};
    const int n = static_cast<int>(p.n);
    const int tail = CubeTailFactorForLength(n);
    if (tail == 0)
        return {};
    const bool blocked = tail >= 129;
    const int prefix = n / tail, half = tail / 2;
    const int width = (half / 16 + 1) * 16;
    if (!blocked && prefix == 32)
        return {tail, 32, width, 1, 1, true};
    const int rowFloats = (2 * n + 511) / 512 * 512;
    CubeTailPlan best;
    int bestWork = 0, bestCalls = 0;
    for (int j : {32, 16, 8}) {
        if (blocked && prefix % j != 0)
            continue;
        for (int tile : {64, 48, 32, 16}) {
            if ((blocked && tile == 16) || (!blocked && tile > width))
                continue;
            const int kb = (half + tile - 1) / tile, nb = (half + tile) / tile;
            // UB holds three rows and six half-matrix fragments. L1 holds two input slots
            // for both AIVs and both (cosine/sine) stages. Weights use a separate L1 region.
            if (12 * rowFloats + 48 * j * tile > 212 * 1024 || 64 * j * kb * tile > 128 * 1024)
                continue;
            const int groups = (prefix + j - 1) / j;
            const int work = groups * j * kb * nb * tile * tile;
            const int calls = groups * kb * nb;
            if (!best.tail || work < bestWork || (work == bestWork && calls < bestCalls)) {
                best = {tail, j, tile, kb, nb, false};
                bestWork = work;
                bestCalls = calls;
            }
        }
    }
    return best;
}
inline bool UseMixedCubeTail(const FftMixedCommon& p) { return SelectCubeTail(p).tail != 0; }
inline int CubeWeightOffset(int n) { return (2 * n + 7) / 8 * 8; }
inline bool ValidMixedCommon(const FftMixedCommon& p)
{
    if (p.complex2dAxis || (p.cubeTailAllowed && p.pass == MixedCommonPass::Complex)) {
        // Standalone C2C retains its existing power-of-two route and batch limit.
        // The rectangular 2D planner separately selects RegBase for power-of-two axes.
        if (p.pass != MixedCommonPass::Complex || p.n < 1 || p.n > 8192 || p.rows < 1 ||
            p.rows > 0x7fffffff / (2 * p.n))
            return false;
        if (!p.complex2dAxis && (p.rows > 8192 || (p.n & (p.n - 1)) == 0))
            return false;
        int64_t left = p.n;
        for (int radix = 2; radix <= 251 && left > 1; ++radix)
            while (left % radix == 0)
                left /= radix;
        return left == 1;
    }
    const bool lengthOk = p.cubeTailAllowed && p.pass != MixedCommonPass::Complex ? RealMixedLength(p.n) :
                                                                                    MixedCommonLength(p.n);
    return lengthOk && p.rows >= 1 && p.rows <= 8192 && int(p.pass) >= 0 && int(p.pass) <= 2 &&
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
    const int stride = ((real ? p.n : 2 * p.n) + 135) / 8 * 8;
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
inline int64_t MixedCommonCoeffFloats(const FftMixedCommon& p)
{
    const auto plan = SelectCubeTail(p);
    if (!plan.tail)
        return 2 * p.n;
    return CubeWeightOffset(p.n) + 2 * plan.width * plan.width * plan.kBlocks * plan.outBlocks;
}
struct MixedCommonTiling {
    int32_t n, rows, pass, inverse, radix2Stages, oddStages, twoFrequencies, rowGroup;
    int32_t radix[8], prev[8];
    int32_t tail, kBlocks, outBlocks, rowsPerAivTile, realCubePipeline;
};
inline bool BuildMixedCommonTiling(const FftMixedCommon& p, MixedCommonTiling& t)
{
    t = {};
    t.rowsPerAivTile = 1;
    if (!ValidMixedCommon(p))
        return false;
    t.n = int32_t(p.n);
    t.rows = int32_t(p.rows);
    // Assign complete 32-byte GM blocks to one core even for short/odd rows.
    t.rowGroup = (p.complex2dAxis || p.cubeTailAllowed) && p.n % 4 != 0 ? (p.n % 2 == 0 ? 2 : 4) : 1;
    t.pass = int32_t(p.pass);
    t.inverse = p.inverse;
    // Use paired symmetric outputs for larger transforms.
    t.twoFrequencies = p.n >= 1824;
    int left = p.pass == MixedCommonPass::Complex ? t.n : t.n / 2, prev = 1;
    const auto cube = SelectCubeTail(p);
    t.rowsPerAivTile = cube.rowsPerAivTile;
    t.realCubePipeline = UseRealCubePipeline(p, cube);
    if (cube.tail && p.pass != MixedCommonPass::Complex)
        t.rowGroup = 4;
    else if (p.pass != MixedCommonPass::Complex && p.cubeTailAllowed && FixedRealTargetTail(p) < 0) {
        // Adjacent AIVs must not write parts of the same 32-byte GM block.
        const int outputFloats = p.pass == MixedCommonPass::Real ? t.n + 2 : t.n;
        t.rowGroup = outputFloats % 8 == 0 ? 1 : outputFloats % 4 == 0 ? 2 : 4;
    }
    t.tail = cube.tail;
    t.kBlocks = cube.kBlocks;
    t.outBlocks = cube.outBlocks;
    if (cube.tail)
        left /= cube.tail;
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
    return t.oddStages > 0 || p.complex2dAxis || cube.tail;
}
} // namespace AsdSip::OpParam
