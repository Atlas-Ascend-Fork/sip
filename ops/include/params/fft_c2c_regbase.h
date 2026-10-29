/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_FFT_C2C_REGBASE_H
#define ASDSIP_PARAMS_FFT_C2C_REGBASE_H

#include <cstdint>
#include <sstream>
#include <string>

namespace AsdSip {
namespace OpParam {
inline bool IsFftC2CRegBaseSupported(int64_t n)
{
    switch (n) {
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
            return true;
        default:
            return false;
    }
}

inline bool IsFftC2CRegBaseOptimizedSupported(int64_t n) { return n >= 32 && n <= 8192 && (n & (n - 1)) == 0; }

// Coefficients occupy N complex FP32 values, ending in zero padding. Original
// and small Optimized kernels use compact radix-2 segments at H-1. Optimized N>=512 uses
// compact radix-4 W/W^2/W^3 planes plus an optional radix-2 final segment.
// The coefficient sign and Optimized radix-4 butterfly specialize direction. Neither
// path normalizes the result.
struct FftC2CRegBase {
    int64_t fftN = 0;
    int64_t batchSize = 0;
    int32_t isInverse = 0;
    int32_t optimizedConfig = 0; // Zero preserves the original C2C kernel and coefficient layout.

    bool operator==(const FftC2CRegBase& other) const
    {
        return fftN == other.fftN && batchSize == other.batchSize && isInverse == other.isInverse &&
               optimizedConfig == other.optimizedConfig;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: FftC2CRegBase, fftN:" << fftN << ", batchSize:" << batchSize << ", isInverse:" << isInverse
           << ", optimizedConfig:" << optimizedConfig;
        return ss.str();
    }
};
} // namespace OpParam
} // namespace AsdSip
#endif
