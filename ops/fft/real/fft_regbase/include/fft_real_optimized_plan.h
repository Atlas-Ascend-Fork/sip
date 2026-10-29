/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "fft_real_plan.h"

namespace fft::real {
inline bool SupportsOptimized(int32_t n, int64_t batch, Kind kind, bool inverse)
{
    return n >= 32 && n <= 8192 && (n & (n - 1)) == 0 && batch == n &&
           ((kind == Kind::R2C && !inverse) || (kind == Kind::C2R && inverse));
}
inline int NormalizeOptimizedConfig(int n, int config)
{
    const int expected = n == 32 || n == 64                 ? 1 :
                         n == 256 || n == 512               ? 6 :
                         n == 128 || n == 1024 || n == 8192 ? 2 :
                         n == 2048                          ? 4 :
                         n == 4096                          ? 3 :
                                                              0;
    if (config == 0 || config != expected)
        throw std::invalid_argument("Unsupported optimized real FFT configuration");
    return config;
}
inline Descriptor SelectOptimized(int32_t n, int64_t batch, Kind kind, bool inverse, uint32_t aivCount, int config)
{
    if (!SupportsOptimized(n, batch, kind, inverse) || aivCount == 0)
        throw std::invalid_argument("real FFT Optimized requires a standard-direction A5 target row batch");
    auto d = Select(n, batch, kind, inverse, aivCount);
    d.optimizedConfig = NormalizeOptimizedConfig(n, config);
    d.variant = Variant::FusedUb;
    const int m = n / 2;
    d.fusedTileBatch = m <= 256 ? 256 / m : (m == 512 ? 2 : 1);
    if (d.optimizedConfig == 3 || d.optimizedConfig == 5)
        d.fusedTileBatch = 2;
    if (d.optimizedConfig == 4)
        d.fusedTileBatch = 4;
    if (d.optimizedConfig == 6 && n == 512)
        d.fusedTileBatch = 4;
    // Keep original row distribution. Only local DMA/compute tile capacity changes.
    d.fusedBlocks = d.core.phaseABlocks;
    d.fusedUbSlots = d.optimizedConfig != 5 && batch > int64_t{d.fusedBlocks} * d.fusedTileBatch ? 2 : 1;
    d.scratchBytes = d.workspaceBytes = d.boundaryBlocks = d.boundaryUbBytes = 0;
    // Only W[0..M/2] is needed by the paired conversion. Pad to a 32-byte transfer.
    d.boundaryTwiddleBytes = size_t(n / 2 + 8) * sizeof(float);
    const size_t packed = m >= 512 ? d.fusedTileBatch * (n + 8U) + 128U : d.fusedTileBatch * n;
    const size_t half = m >= 512 ? packed : d.fusedTileBatch * (n + 8U);
    d.fusedUbBytes = sizeof(float) * (d.fusedUbSlots * (packed + half) + n + n / 2 + 8U);
    if (d.fusedUbBytes > c2c::kUserUbBytes)
        throw std::logic_error("real FFT Optimized UB budget exceeded");
    return d;
}
inline std::vector<float> BuildOptimizedBoundaryTwiddles(const Descriptor& d)
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<float> table(d.boundaryTwiddleBytes / sizeof(float), 0.0f);
    for (int k = 0; k <= d.n / 4; ++k) {
        const double angle = (d.inverse ? 1.0 : -1.0) * 2.0 * pi * k / d.n;
        table[2 * k] = static_cast<float>(std::cos(angle));
        table[2 * k + 1] = static_cast<float>(std::sin(angle));
    }
    table[0] = 1.0f;
    table[1] = 0.0f;
    // The midpoint has an exact +/-i coefficient.
    table[d.n / 2] = 0.0f;
    table[d.n / 2 + 1] = d.inverse ? 1.0f : -1.0f;
    return table;
}
} // namespace fft::real
