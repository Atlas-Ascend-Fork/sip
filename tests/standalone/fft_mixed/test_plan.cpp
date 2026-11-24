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
#include <stdexcept>
#include "fft_mixed_selection.h"
using namespace AsdSip;
void require(bool b)
{
    if (!b)
        throw std::runtime_error("host contract failed");
}
int main()
{
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
    for (int n = 1; n <= 8192; ++n) {
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
    for (auto n : {393, 637})
        require(fft1d::CommonMixedEligible(n, 17, 0, true, true, true, true));
    for (auto n : {0, 128, 257, 8193})
        require(!fft1d::CommonMixedEligible(n, 17, 0, false, true, true, true));
    for (auto b : {0, 8193})
        require(!fft1d::CommonMixedEligible(465, b, 0, false, true, true, true));
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
    std::cout << "HOST_PASS positive_grid=" << count << " cube_axis_lengths=8192\n";
}
