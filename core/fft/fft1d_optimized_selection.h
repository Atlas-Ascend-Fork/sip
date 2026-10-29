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
#include <cstdint>

namespace AsdSip::fft1d {
enum class Kind { C2C, R2C, C2R, Other };
inline int OptimizedConfig(int64_t n, int64_t batch, Kind kind, bool inverse, bool ascend950, bool horizontal)
{
    if (!ascend950 || !horizontal || n < 32 || n > 8192 || (n & (n - 1)) != 0 || batch != n)
        return 0;
    if (kind == Kind::C2C)
        return 1;
    if ((kind != Kind::R2C && kind != Kind::C2R) || (kind == Kind::C2R) != inverse)
        return 0;
    switch (n) {
        case 32:
        case 64:
            return 1;
        case 128:
            return 2;
        case 256:
        case 512:
            return 6;
        case 1024:
        case 8192:
            return 2;
        case 2048:
            return 4;
        case 4096:
            return 3;
        default:
            return 0;
    }
}
// The existing 2D planner transposes columns into contiguous row batches.
// Real transforms have N/2+1 complex columns; the real axis still has N rows.
inline int OptimizedSquareConfig(int64_t n, int64_t otherAxis, int64_t planBatch, int64_t axisBatch, Kind axisKind,
                                 Kind planKind, bool inverse, bool ascend950, bool horizontal)
{
    if (n != otherAxis || planBatch != 1 || planKind == Kind::Other)
        return 0;
    if ((planKind == Kind::R2C && inverse) || (planKind == Kind::C2R && !inverse))
        return 0;
    const bool complexAxis = axisKind == Kind::C2C;
    if (!complexAxis && axisKind != planKind)
        return 0;
    const int64_t expectedBatch = complexAxis && planKind != Kind::C2C ? n / 2 + 1 : n;
    if (axisBatch != expectedBatch)
        return 0;
    return OptimizedConfig(n, n, axisKind, inverse, ascend950, horizontal);
}
} // namespace AsdSip::fft1d
