/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <stdexcept>
#include "fft_mixed_selection.h"
using namespace AsdSip;
void require(bool b)
{
    if (!b)
        throw std::runtime_error("host contract failed");
}
int main(int argc, char** argv)
{
    using Pass = OpParam::MixedCommonPass;
    using Route = fft1d::Real2dAxisRoute;
    require(fft1d::SelectReal2dAxis(251, 251, Pass::Real, false, true) == Route::Native);
    require(fft1d::SelectReal2dAxis(64, 6000, Pass::Complex, false) == Route::Native);
    require(fft1d::SelectReal2dAxis(58, 37, Pass::Real, false) == Route::Native);
    require(fft1d::SelectReal2dAxis(64, 128, Pass::Hermitian, true) == Route::RegBase);
    require(fft1d::SelectReal2dAxis(64, 1024, Pass::Complex, false) == Route::RegBase);
    require(fft1d::SelectReal2dAxis(16, 1984, Pass::Complex, false) == Route::Mixed);
    require(fft1d::SelectReal2dAxis(490, 512, Pass::Real, false) == Route::Mixed);
    require(fft1d::SelectReal2dAxis(490, 513, Pass::Real, false) == Route::Native);
    require(fft1d::SelectReal2dAxis(875, 256, Pass::Hermitian, true) == Route::Native);
    require(fft1d::SelectReal2dAxis(928, 928, Pass::Real, false, true) == Route::Mixed);
    require(fft1d::SelectReal2dAxis(128, 193, Pass::Real, false) == Route::RegBase);
    for (auto pass : {Pass::Real, Pass::Hermitian}) {
        for (bool inverse : {false, true}) {
            const auto route = [pass, inverse](int n, int rows) {
                return fft1d::SelectReal2dAxis(n, rows, pass, inverse);
            };
            require(route(45, 128) == Route::Mixed && route(45, 129) == Route::Native);
            require(route(49, 64) == Route::Native && route(35, 31) == Route::Native);
            require(route(48, 512) == Route::Mixed && route(48, 513) == Route::Native);
            require(route(70, 128) == Route::Native && route(110, 128) == Route::Native);
            require(route(350, 127) == Route::Native && route(350, 128) == Route::Mixed);
            require(route(1008, 512) == Route::Mixed && route(1008, 513) == Route::Native);
        }
    }
    require(fft1d::SelectReal2dAxis(32, 31, Pass::Complex, false) == Route::Native);
    require(fft1d::SelectReal2dAxis(32, 32, Pass::Complex, false) == Route::RegBase);
    require(fft1d::SelectReal2dAxis(32, 4096, Pass::Complex, true) == Route::RegBase);
    require(fft1d::SelectReal2dAxis(32, 4097, Pass::Complex, true) == Route::Native);
    for (auto pass : {Pass::Real, Pass::Hermitian}) {
        for (bool inverse : {false, true}) {
            for (int n : {100, 135, 928, 1037, 1573, 5400, 7007}) {
                OpParam::FftMixedCommon p{n, 8201, pass, inverse, false, true, true, true};
                OpParam::MixedCommonTiling t{};
                if (n > 1024)
                    require(fft1d::SelectReal2dAxis(n, p.rows, pass, inverse) == Route::Mixed);
                require(OpParam::BuildMixedCommonTiling(p, t));
                if (n % 2)
                    require(t.tail == 0 && t.rowGroup == 8);
                const int spectrum = 2 * (n / 2 + 1);
                require(OpParam::MixedCommonFloats(p, pass == Pass::Real) == spectrum * p.rows);
                require(OpParam::MixedCommonFloats(p, pass != Pass::Real) == n * p.rows);
                int factors = (1 << t.radix2Stages) * (t.tail ? t.tail : 1);
                for (int stage = 0; stage < t.oddStages; ++stage)
                    factors *= t.radix[stage];
                require(factors == (n % 2 ? n : n / 2));
                require(t.rowGroup * (pass == Pass::Real ? spectrum : n) % 8 == 0);
            }
        }
    }
    require(fft1d::SelectReal2dAxis(5400, 200000, Pass::Real, false) == Route::Native);
    require(!OpParam::ValidMixedCommon({257, 10, Pass::Hermitian, true, false, true, true, true}));
    // All eligible lengths must fit their selected dynamic layout; boundaries
    // change the number of buffers, never fall back just because rows == 1.
    for (int n = 32; n <= 8192; n += 32) {
        if (!OpParam::MixedCommonLength(n))
            continue;
        for (auto pass :
             {OpParam::MixedCommonPass::Complex, OpParam::MixedCommonPass::Real, OpParam::MixedCommonPass::Hermitian}) {
            OpParam::FftMixedCommon p{n, 193, pass, pass == OpParam::MixedCommonPass::Hermitian};
            auto plan = OpParam::SelectMixedCommonTile(p);
            require(plan.rows >= 1 && plan.buffers >= 2 && plan.buffers <= 4);
            require(plan.buffers * plan.capacity + 2 * n <= 54272 && plan.capacity % 8 == 0);
            const int stride = (pass == OpParam::MixedCommonPass::Complex ? 2 * n : n) + 128;
            require(plan.rows * stride <= plan.capacity && (plan.rows + 1) * stride > plan.capacity);
            if (pass == OpParam::MixedCommonPass::Complex)
                require(plan.key == (n <= 2944 ? 19 : n <= 6720 ? 22 : 26));
            else
                require(plan.key == (n > 5312 && n <= 6688 ? 25 : 21));
            // Key 26 computes one complex row, so its DMA tile must also contain exactly one row.
            if (plan.key == 26)
                require(pass == OpParam::MixedCommonPass::Complex && plan.rows == 1);
            p.rows = 0;
            require(OpParam::SelectMixedCommonTile(p).key == -1);
            require(OpParam::SelectMixedCommonTile(p).buffers == 0);
        }
    }
    for (int n : {16, 60, 81, 875, 1573, 5400}) {
        OpParam::FftMixedCommon p{n, 16384, OpParam::MixedCommonPass::Complex, false, true};
        OpParam::MixedCommonTiling tiling{};
        require(OpParam::BuildMixedCommonTiling(p, tiling));
        require(OpParam::SelectMixedCommonTile(p).rows > 0);
        int product = 1 << tiling.radix2Stages;
        for (int i = 0; i < tiling.oddStages; ++i)
            product *= tiling.radix[i];
        require(product == n);
    }
    require(!OpParam::ValidMixedCommon({257, 32, OpParam::MixedCommonPass::Complex, false, true}));
    struct TailCase {
        int n, tail;
    };
    // These off-grid lengths continue to use the general prime-factor strategy.
    const TailCase cases[] = {{16 * 23, 0},    {16 * 29, 29},   {16 * 49, 0},    {16 * 51, 51},   {16 * 89, 89},
                              {16 * 105, 0},   {16 * 119, 119}, {16 * 121, 121}, {16 * 123, 123}, {16 * 127, 127},
                              {16 * 129, 43},  {16 * 215, 43},  {16 * 217, 31},  {16 * 255, 85},  {16 * 251, 251},
                              {24 * 131, 131}, {465, 93},       {393, 0},        {637, 0},        {257, 0},
                              {133, 0},        {161, 0}};
    for (const auto& c : cases) {
        OpParam::FftMixedCommon p{c.n, 3, OpParam::MixedCommonPass::Complex, false, true, true};
        require(OpParam::SelectCubeTail(p).tail == c.tail);
        p.inverse = true;
        require(OpParam::SelectCubeTail(p).tail == c.tail);
        p.cubeTailAllowed = false;
        require(!OpParam::UseMixedCubeTail(p));
    }
    // Include small and blocked Cube tails, the tied route, and a measured Vec winner.
    const TailCase targets[] = {{736, 23}, {1760, 55}, {5440, 85}, {8032, 251}, {8160, 0}};
    for (const auto& c : targets) {
        OpParam::FftMixedCommon p{c.n, c.n, OpParam::MixedCommonPass::Complex, false, false, true};
        require(OpParam::SelectCubeTail(p).tail == c.tail);
    }
    int hostCoverageCases = 0;
    for (int n = 1; n <= 8192; ++n) {
        const bool power2 = (n & (n - 1)) == 0;
        for (int batch : {1, 16, 256}) {
            for (bool inverse : {false, true}) {
                OpParam::FftMixedCommon axis{n, batch, Pass::Complex, inverse, false, true};
                OpParam::MixedCommonTiling tiling{};
                require(fft1d::CommonMixedEligible(n, batch, 0, inverse, true, true, true) == !power2);
                require(OpParam::BuildMixedCommonTiling(axis, tiling) == !power2);
                if (power2) {
                    // A power-of-two length also has a native fallback when optimized RegBase is bypassed.
                    const auto native = fft::a5::Select(n, batch);
                    require(native == fft::a5::Route::Regbase || native == fft::a5::Route::Dft);
                } else {
                    int product = 1 << tiling.radix2Stages;
                    for (int stage = 0; stage < tiling.oddStages; ++stage) {
                        require(tiling.prev[stage] == product);
                        product *= tiling.radix[stage];
                    }
                    require(product * (tiling.tail ? tiling.tail : 1) == n);
                    require(OpParam::SelectMixedCommonTile(axis).rows > 0);
                }
                ++hostCoverageCases;
            }
        }
        OpParam::FftMixedCommon p{n, 17, OpParam::MixedCommonPass::Complex, false, false, true};
        const auto cube = OpParam::SelectCubeTail(p);
        if (n % 32 == 0) {
            auto axis = p;
            axis.inverse = true;
            require(OpParam::SelectCubeTail(axis).tail == cube.tail);
            axis.complex2dAxis = true;
            axis.rows = 9173;
            require(OpParam::SelectCubeTail(axis).tail == cube.tail);
            axis.pass = OpParam::MixedCommonPass::Real;
            require(!OpParam::UseMixedCubeTail(axis));
        }
        OpParam::MixedCommonTiling t{};
        const bool eligible = OpParam::BuildMixedCommonTiling(p, t);
        require(eligible == fft1d::CommonMixedEligible(n, 17, 0, false, true, true, true));
        if (eligible) {
            int prefix = 1 << t.radix2Stages;
            for (int i = 0; i < t.oddStages; ++i) {
                require(t.prev[i] == prefix);
                prefix *= t.radix[i];
            }
            require(prefix * (cube.tail ? cube.tail : 1) == n);
        }
        require(cube.rowsPerAivTile == 1 && t.rowsPerAivTile == 1);
        if (!cube.tail)
            continue;
        require(eligible && t.tail == cube.tail && t.outBlocks == cube.outBlocks && t.kBlocks == cube.kBlocks);
        require(OpParam::MixedCommonCoeffFloats(p) ==
                OpParam::CubeWeightOffset(n) + 2 * cube.width * cube.width * cube.kBlocks * cube.outBlocks);
        require(OpParam::CubeWeightOffset(n) % 8 == 0);
        if (cube.direct) {
            require(n / cube.tail == 32 && cube.tail < 129);
        } else {
            const int rowFloats = (2 * n + 511) / 512 * 512;
            require(12 * rowFloats + 48 * cube.lanes * cube.width <= 212 * 1024);
            require(64 * cube.lanes * cube.kBlocks * cube.width <= 128 * 1024);
        }
    }
    for (auto n : {257, 393, 637, 514, 1799, 8191})
        require(fft1d::CommonMixedEligible(n, 17, 0, true, true, true, true));
    for (auto n : {0, 128, 8193})
        require(!fft1d::CommonMixedEligible(n, 17, 0, false, true, true, true));
    require(!fft1d::CommonMixedEligible(465, 0, 0, false, true, true, true));
    for (int n : {511, 1023}) {
        const bool inverse = n == 511;
        const int dtsRows = inverse ? 261121 : 260865;
        const int maxRows = INT32_MAX / (2 * n);
        for (int rows : {8192, 8193, dtsRows, maxRows}) {
            require(fft1d::CommonMixedEligible(n, rows, 0, inverse, true, true, true));
            OpParam::MixedCommonTiling t{};
            require(OpParam::BuildMixedCommonTiling({n, rows, Pass::Complex, inverse, false, true}, t));
            require(t.rows == rows && t.n == n && t.inverse == inverse);
        }
        require(!fft1d::CommonMixedEligible(n, maxRows + 1, 0, inverse, true, true, true));
    }
    int count = 0;
    for (int k = 1; k <= 256; ++k)
        for (int pass = 0; pass < 3; ++pass) {
            const int n = 32 * k;
            const bool inv = pass == 2;
            OpParam::FftMixedCommon p{n, 7, static_cast<OpParam::MixedCommonPass>(pass), inv};
            OpParam::MixedCommonTiling t{};
            bool mixed = (n & (n - 1)) != 0;
            require(OpParam::BuildMixedCommonTiling(p, t) == mixed);
            require(fft1d::CommonMixedEligible(n, 7, pass, inv, true, true) == (mixed && (pass == 0 || n > 1024)));
            if (mixed) {
                int product = 1 << t.radix2Stages;
                for (int i = 0; i < t.oddStages; ++i) {
                    require(t.prev[i] == product);
                    product *= t.radix[i];
                }
                require(product == (pass == 0 ? n : n / 2));
                require(t.twoFrequencies == (n >= 1824));
            }
            ++count;
        }
    for (auto n : {0, 31, 33, 8193, 16384})
        require(!fft1d::CommonMixedEligible(n, 1, 0, false, true, true));
    for (auto b : {0, 8193})
        require(!fft1d::CommonMixedEligible(352, b, 0, false, true, true));
    require(!fft1d::CommonMixedEligible(352, 1, 0, false, false, true));
    require(!fft1d::CommonMixedEligible(352, 1, 0, false, true, false));
    require(!fft1d::CommonMixedEligible(1536, 1, 1, true, true, true));
    require(!fft1d::CommonMixedEligible(1536, 1, 2, false, true, true));
    int realCases = 0, realScanCases = 0;
    for (int n = 1; n <= 8192; ++n) {
        const bool odd = n % 2 != 0;
        const int complexSize = odd ? n : n / 2;
        const bool publicEligible = n >= 96 && (n & (n - 1)) != 0;
        require(OpParam::RealMixedLength(n) == publicEligible);
        for (int batch : {1, 3, 7, 8, 16, 193, 256, 8192}) {
            for (auto pass : {OpParam::MixedCommonPass::Real, OpParam::MixedCommonPass::Hermitian}) {
                const bool inverse = pass == OpParam::MixedCommonPass::Hermitian;
                OpParam::FftMixedCommon p{n, batch, pass, inverse, false, true};
                const auto cube = OpParam::SelectCubeTail(p);
                OpParam::MixedCommonTiling t{};
                const bool valid = OpParam::BuildMixedCommonTiling(p, t);
                require(t.rowsPerAivTile >= 1);
                require(fft1d::CommonMixedEligible(n, batch, int(pass), inverse, true, true,
                                                   OpParam::RealMixedLength(n)) == publicEligible);
                require(valid == publicEligible);
                ++realScanCases;
                if (batch == 1 || batch == 16 || batch == 256)
                    ++hostCoverageCases;
                if (!valid) {
                    require(cube.tail == 0);
                    const auto native = fft::a5::SelectReal(n, batch, inverse);
                    require(native == fft::a5::RealRoute::Dft || native == fft::a5::RealRoute::Regbase);
                    continue;
                }
                const int spectrumFloats = 2 * (n / 2 + 1);
                const int outputFloats = inverse ? n : spectrumFloats;
                require(OpParam::MixedCommonFloats(p, !inverse) == p.rows * spectrumFloats);
                require(OpParam::MixedCommonFloats(p, inverse) == p.rows * n);
                if (odd)
                    require(t.rowGroup == 8 && t.rowsPerAivTile == 1);
                if (OpParam::FixedRealTargetTail(p) < 0)
                    require((t.rowGroup * outputFloats) % 8 == 0);
                int prefix = 1 << t.radix2Stages;
                for (int s = 0; s < t.oddStages; ++s) {
                    require(t.prev[s] == prefix && t.radix[s] >= 3 && t.radix[s] <= complexSize);
                    prefix *= t.radix[s];
                }
                require(prefix * (t.tail ? t.tail : 1) == complexSize);
                require(t.tail == cube.tail);
                const auto tile = OpParam::SelectMixedCommonTile(p);
                const int stride = (2 * complexSize + 135) / 8 * 8;
                require(tile.rows > 0 && tile.rows * stride <= tile.capacity);
                require(tile.buffers * tile.capacity + 2 * n <= 54272);
                if (!cube.tail) {
                    require(t.rowsPerAivTile == 1 && t.realCubePipeline == 0);
                    continue;
                }
                require(!cube.direct && t.rowGroup == (odd ? 8 : 4));
                require(cube.rowsPerAivTile == 1 || cube.rowsPerAivTile == 2);
                require(t.rowsPerAivTile == cube.rowsPerAivTile);
                const int rootFloats = (2 * n + 511) / 512 * 512;
                const int rowFloats = ((odd ? 2 * n : n + 128) + 511) / 512 * 512;
                const int pipelineBytes = t.realCubePipeline ? 16 * cube.lanes : 0;
                require(4 * (rootFloats + 2 * cube.rowsPerAivTile * rowFloats) + 48 * cube.lanes * cube.width +
                            pipelineBytes <=
                        212 * 1024);
                require(64 * cube.lanes * cube.kBlocks * cube.width <= 128 * 1024);
                require(OpParam::MixedCommonCoeffFloats(p) ==
                        OpParam::CubeWeightOffset(n) + 2 * cube.width * cube.width * cube.kBlocks * cube.outBlocks);
                if (t.realCubePipeline) {
                    require(cube.kBlocks == 1 && cube.outBlocks == 1 && cube.rowsPerAivTile * prefix <= cube.lanes);
                    p.allowRealCubePipeline = false;
                    require(!OpParam::UseRealCubePipeline(p, cube));
                }
                if (cube.rowsPerAivTile == 2)
                    require(prefix == 16 && cube.lanes == 32 && batch >= 8);
                ++realCases;
                p.inverse = !inverse;
                require(!OpParam::UseMixedCubeTail(p));
                p.inverse = inverse;
                p.cubeTailAllowed = false;
                require(!OpParam::UseMixedCubeTail(p));
            }
        }
    }
    for (auto pass : {Pass::Real, Pass::Hermitian}) {
        struct OddTileCase {
            int n, buffers, rows;
        };
        const OddTileCase boundaries[] = {{2943, 4, 2}, {2945, 3, 2}, {3819, 3, 2},
                                          {3825, 3, 1}, {6731, 3, 1}, {6741, 2, 1}};
        for (const auto& c : boundaries) {
            OpParam::FftMixedCommon p{c.n, 16, pass, pass == Pass::Hermitian, false, true};
            const auto tile = OpParam::SelectMixedCommonTile(p);
            require(tile.buffers == c.buffers && tile.rows == c.rows);
        }
        const TailCase batchBoundaries[] = {{96, 0}, {1037, 61}, {1081, 47}, {1215, 0}, {2848, 89}};
        for (const auto& c : batchBoundaries) {
            const int n = c.n;
            const int maxRows = INT32_MAX / (2 * (n / 2 + 1));
            OpParam::FftMixedCommon p{n, 8192, pass, pass == Pass::Hermitian, false, true};
            for (int rows : {0, 8192, 8193, 260865, maxRows, maxRows + 1}) {
                p.rows = rows;
                const bool fits = rows > 0 && rows <= maxRows;
                require(fft1d::CommonMixedEligible(n, rows, int(pass), p.inverse, true, true, true) == fits);
                OpParam::MixedCommonTiling t{};
                require(OpParam::BuildMixedCommonTiling(p, t) == fits);
                require(OpParam::UseMixedCubeTail(p) == (fits && c.tail != 0));
                if (fits)
                    require(t.rows == rows && t.pass == int(pass) && t.tail == c.tail);
            }
        }
    }
    for (int n : {0, 514, 1024, 8193})
        require(!OpParam::UseMixedCubeTail({n, 8192, OpParam::MixedCommonPass::Real, false, false, true}));
    const TailCase realFallback[] = {{1424, 89}, {2096, 131}, {1057, 151}, {262, 0}, {1050, 0}, {4128, 129}, {928, 29}};
    for (const auto& c : realFallback) {
        for (auto pass : {OpParam::MixedCommonPass::Real, OpParam::MixedCommonPass::Hermitian}) {
            OpParam::FftMixedCommon p{c.n, 193, pass, pass == OpParam::MixedCommonPass::Hermitian, false, true};
            require(OpParam::FixedRealTargetTail(p) == -1 && OpParam::SelectCubeTail(p).tail == c.tail);
            require(fft1d::CommonMixedEligible(c.n, p.rows, int(pass), p.inverse, true, true, true));
            require(!fft1d::CommonMixedEligible(c.n, p.rows, int(pass), !p.inverse, true, true, true));
        }
    }
    for (int n : {58, 94})
        require(!fft1d::CommonMixedEligible(n, 193, 1, false, true, true, true));
    // Validate the actual public entry, capacity and kernel schedule for both directional tables.
    for (int n = 32; n <= 8192; n += 32) {
        for (auto pass : {OpParam::MixedCommonPass::Real, OpParam::MixedCommonPass::Hermitian}) {
            OpParam::FftMixedCommon p{n, n, pass, pass == OpParam::MixedCommonPass::Hermitian, false, true};
            const int tail = OpParam::FixedRealTargetTail(p);
            if (!OpParam::MixedCommonLength(n)) {
                require(tail == -1);
                continue;
            }
            require(tail >= 0 && fft1d::CommonMixedEligible(n, n, int(pass), p.inverse, true, true, true));
            const auto cube = OpParam::SelectCubeTail(p);
            OpParam::MixedCommonTiling t{};
            require(OpParam::BuildMixedCommonTiling(p, t) && t.tail == tail && cube.tail == tail);
            int prefix = 1 << t.radix2Stages;
            for (int stage = 0; stage < t.oddStages; ++stage)
                prefix *= t.radix[stage];
            require(prefix * (t.tail ? t.tail : 1) == n / 2);
            if (!t.tail) {
                require(t.rowGroup == 1);
                continue;
            }
            const bool whole = cube.kBlocks == 1 && cube.outBlocks == 1;
            const int ub = OpParam::RealCubeUbBytes(n, cube.rowsPerAivTile, cube.lanes, cube.width);
            require(ub + (t.realCubePipeline == 1 ? 16 * cube.lanes : 0) <= 212 * 1024);
            require(64 * cube.lanes * cube.kBlocks * cube.width <= 128 * 1024);
            if (t.realCubePipeline == 1) {
                require(whole && cube.rowsPerAivTile * prefix <= cube.lanes);
                p.allowRealCubePipeline = false;
                require(OpParam::BuildMixedCommonTiling(p, t) && t.realCubePipeline == 0);
            }
        }
    }
    if (argc > 1) {
        std::ofstream scan(argv[1]);
        require(bool(scan));
        scan << "n,batch,direction,route,candidate_tail,prefix,tail,rows_per_aiv_tile,j,t,k_blocks,out_blocks,cube_ub_"
                "bytes,cube_l1_bytes,"
                "tiling_key,real_kernel_schedule\n";
        for (int n = 32; n <= 8192; n += 32) {
            for (auto pass : {OpParam::MixedCommonPass::Real, OpParam::MixedCommonPass::Hermitian}) {
                OpParam::FftMixedCommon p{n, n, pass, pass == OpParam::MixedCommonPass::Hermitian, false, true};
                const auto cube = OpParam::SelectCubeTail(p);
                const auto candidate = OpParam::SelectRealCubeCandidate(p);
                const bool power2 = (n & (n - 1)) == 0;
                const bool mixed = fft1d::CommonMixedEligible(n, n, int(pass), p.inverse, true, true, true);
                const char* route = power2 ? "power2" : !mixed ? "dft" : cube.tail ? "cube" : "mixed_vec";
                const int rootFloats = (2 * n + 511) / 512 * 512, rowFloats = (n + 128 + 511) / 512 * 512;
                const int ub = cube.tail ? 4 * (rootFloats + 2 * cube.rowsPerAivTile * rowFloats) +
                                               48 * cube.lanes * cube.width :
                                           0;
                const int l1 = cube.tail ? 64 * cube.lanes * cube.kBlocks * cube.width : 0;
                scan << n << ',' << n << ',' << (pass == OpParam::MixedCommonPass::Real ? "r2c" : "c2r") << ',' << route
                     << ',' << candidate.tail << ',' << (candidate.tail ? n / (2 * candidate.tail) : 0) << ','
                     << cube.tail << ',' << cube.rowsPerAivTile << ',' << cube.lanes << ',' << cube.width << ','
                     << cube.kBlocks << ',' << cube.outBlocks << ',' << ub << ',' << l1 << ','
                     << (cube.tail ? cube.TilingKey() : 0) << ',' << OpParam::UseRealCubePipeline(p, cube) << '\n';
            }
        }
        require(bool(scan));
    }
    require(hostCoverageCases == 8192 * 3 * 4);
    std::cout << "HOST_PASS positive_grid=" << count << " cube_axis_lengths=8192 real_scan_cases=" << realScanCases
              << " real_cube_cases=" << realCases << " host_1d_route_cases=" << hostCoverageCases << '\n';
}
