/*
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
#include "core/fft/fft1d_optimized_selection.h"
#include "ops/fft/c2c/fft_regbase/fft_regbase/optimized/fft_c2c_regbase_optimized_plan.h"
#include "fft_c2c_coefficients.h"
#include "ops/fft/real/fft_regbase/include/fft_real_optimized_plan.h"

void Require(bool result)
{
    if (!result)
        throw std::runtime_error("FFT selection/tiling invariant failed");
}
int main()
{
    using AsdSip::fft1d::Kind;
    using AsdSip::fft1d::OptimizedConfig;
    size_t count = 0;
    for (int n = 1; n <= 16384; ++n) {
        for (auto kind : {Kind::C2C, Kind::R2C, Kind::C2R, Kind::Other}) {
            for (bool inverse : {false, true}) {
                for (int batch : {1, n - 1, n, n + 1}) {
                    const int config = OptimizedConfig(n, batch, kind, inverse, true, true);
                    const bool target = n >= 32 && n <= 8192 && (n & (n - 1)) == 0 && batch == n;
                    const bool real = (kind == Kind::R2C && !inverse) || (kind == Kind::C2R && inverse);
                    Require((config != 0) == (target && (kind == Kind::C2C || real)));
                    Require(OptimizedConfig(n, batch, kind, inverse, false, true) == 0);
                    Require(OptimizedConfig(n, batch, kind, inverse, true, false) == 0);
                    if (config) {
                        for (unsigned cores : {1U, 7U, 32U, 64U, 128U}) {
                            if (kind == Kind::C2C) {
                                const auto d = fft::c2c::SelectRegBaseOptimized(n, batch, inverse, cores, config);
                                Require(d.core.phaseAUbBytes <= fft::c2c::kUserUbBytes && d.core.phaseABlocks > 0);
                            } else {
                                const auto k = kind == Kind::R2C ? fft::real::Kind::R2C : fft::real::Kind::C2R;
                                const auto d = fft::real::SelectOptimized(n, batch, k, inverse, cores, config);
                                Require(d.IsFused() && d.fusedUbBytes <= fft::c2c::kUserUbBytes && d.fusedBlocks > 0);
                                Require(fft::real::BuildOptimizedBoundaryTwiddles(d).size() * sizeof(float) ==
                                        d.boundaryTwiddleBytes);
                            }
                        }
                    }
                    ++count;
                }
            }
        }
    }
    for (int n : {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        for (bool inverse : {false, true}) {
            const auto reference = fft::c2c::BuildLocalTwiddles(
                fft::c2c::SelectRegBaseOptimized(n, n, inverse, 1, 1).core);
            Require(reference.size() == static_cast<size_t>(2 * n));
            // The cache key intentionally shares these tables across 1D and real-2D plans.
            for (int rows : {n, n / 2 + 1}) {
                for (unsigned cores : {1U, 7U, 32U, 64U, 128U}) {
                    const auto d = fft::c2c::SelectRegBaseOptimized(n, rows, inverse, cores, 1);
                    Require(fft::c2c::BuildLocalTwiddles(d.core) == reference);
                }
            }
            for (auto planKind : {Kind::C2C, inverse ? Kind::C2R : Kind::R2C}) {
                for (auto kind : {Kind::C2C, planKind}) {
                    const int batch = kind == Kind::C2C && planKind != Kind::C2C ? n / 2 + 1 : n;
                    const auto select = [&](int other, int b, int rows, bool platform, bool contiguous) {
                        return AsdSip::fft1d::OptimizedSquareConfig(n, other, b, rows, kind, planKind, inverse,
                                                                    platform, contiguous);
                    };
                    const int config = select(n, 1, batch, true, true);
                    Require(config > 0);
                    Require(select(n + 32, 1, batch, true, true) == 0);
                    Require(select(n, 2, batch, true, true) == 0);
                    Require(select(n, 1, batch + 1, true, true) == 0);
                    Require(select(n, 1, batch, false, true) == 0);
                    Require(select(n, 1, batch, true, false) == 0);
                    if (kind == Kind::C2C) {
                        for (unsigned cores : {1U, 7U, 32U, 64U, 128U}) {
                            const auto d = fft::c2c::SelectRegBaseOptimized(n, batch, inverse, cores, config);
                            Require(d.core.phaseAUbBytes <= fft::c2c::kUserUbBytes);
                        }
                    }
                }
            }
        }
    }
    std::cout << "PASS " << count << " selection cases and target tiling checks\n";
}
