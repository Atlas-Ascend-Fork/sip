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
#include "../large/fft_c2c_plan.h"

namespace fft::c2c {
struct RegBaseOptimizedDescriptor {
    Descriptor core;
    int config = 1;
    bool doubleBuffer = false;
    uint64_t tilingKey = 0;
};

inline int CanonicalRegBaseOptimizedConfig(int32_t n, int config)
{
    if (config != 1)
        throw std::invalid_argument("C2C RegBase Optimized config must be 1");
    (void)n;
    return 1;
}

inline RegBaseOptimizedDescriptor SelectRegBaseOptimized(int32_t n, int64_t batch, bool inverse, uint32_t aivCount,
                                                         int config)
{
    if (n < 32 || n > 8192 || (n & (n - 1)) != 0)
        throw std::invalid_argument("C2C RegBase Optimized requires a power-of-two size in [32,8192]");
    RegBaseOptimizedDescriptor d;
    d.config = CanonicalRegBaseOptimizedConfig(n, config);
    d.core = Select(n, batch, inverse, aivCount);
    // Keep the original public C2C tile, rather than the standalone selector's
    // separate N=256/high-batch override. Every value must match a compiled key.
    d.core.tilePoints = n <= 256 ? 256 : std::max(1024, n);
    const int32_t tileBatch = d.core.tilePoints / n;
    d.doubleBuffer = n <= 4096 && batch > int64_t{d.core.phaseABlocks} * tileBatch;
    if (n <= 64) {
        d.core.smallVariant = d.doubleBuffer ? SmallVariant::FusedRegisterDirectDoubleBuffer :
                                               SmallVariant::FusedRegisterDirect;
    } else if (n <= 256) {
        // The same register-resident body supports either external DMA schedule.
        d.core.smallVariant = SmallVariant::FusedRegisterLazyContiguousDirectDoubleBuffer;
    } else {
        d.core.smallVariant = d.doubleBuffer ? SmallVariant::FusedRadix4UnrolledDirectDoubleBuffer :
                                               SmallVariant::FusedRadix4UnrolledDirect;
    }
    const bool resident = n <= 256;
    d.core.phaseAUbBytes = size_t{d.doubleBuffer ? 2U : 1U} * (resident ? 1U : 2U) * size_t{8} *
                               (d.core.tilePoints + (resident ? 0 : 64)) +
                           d.core.twiddleBytes;
    if (d.core.phaseAUbBytes > kUserUbBytes)
        throw std::logic_error("C2C RegBase Optimized UB budget exceeded");
    // Register variants encode direction entirely in their radix-2 twiddles.
    // Radix-4 additionally specializes its +/-i rotation by direction.
    d.tilingKey = 2000000ULL + static_cast<uint64_t>(n) * 16 + static_cast<uint64_t>(d.config - 1) * 4 +
                  (d.doubleBuffer ? 2 : 0) + (n >= 512 && inverse ? 1 : 0);
    return d;
}

inline const char* RegBaseOptimizedVariantName(const RegBaseOptimizedDescriptor& d)
{
    return d.core.n <= 64 ? "register" : d.core.n <= 256 ? "lazy-contiguous" : "radix4";
}
} // namespace fft::c2c
