/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_KAISER_WINDOW_KERNEL_H
#define ASDSIP_KAISER_WINDOW_KERNEL_H
#include "kernel_common.h"

namespace KaiserWindow {
struct Arch22Compute {
    __aicore__ static inline void MakeNumerator(LocalTensor<int32_t> indices, int32_t first, uint32_t count)
    {
        // Seed one block with scalar stores, then double the initialized prefix.
        // This takes O(log(tile size)) vector instructions, rather than one
        // instruction per 64-element repeat of the general-purpose Arange API.
        auto vectorToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
        const uint32_t seed = count < 8 ? count : 8;
        for (uint32_t i = 0; i < seed; ++i) {
            indices.SetValue(i, first + static_cast<int32_t>(2 * i));
        }
        auto scalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
        for (uint32_t initialized = 8; initialized < count; initialized *= 2) {
            const uint32_t tail = count - initialized;
            const uint32_t chunk = tail < initialized ? tail : initialized;
            Adds(indices[initialized], indices, static_cast<int32_t>(2 * initialized), chunk);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ static inline void Horner(LocalTensor<float> value, LocalTensor<float> variable, float coefficient,
                                         uint32_t count)
    {
        Mul(value, value, variable, count);
        PipeBarrier<PIPE_V>();
        Adds(value, value, coefficient, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ static inline void SmallI0e(LocalTensor<float> x, LocalTensor<float> y, LocalTensor<float> temporary,
                                           LocalTensor<float> result, uint32_t count)
    {
        Mins(y, x, 3.75F, count);
        PipeBarrier<PIPE_V>();
        Muls(y, y, 1.0F / 3.75F, count);
        PipeBarrier<PIPE_V>();
        Mul(y, y, y, count);
        Duplicate(result, 0.0045813F, count);
        PipeBarrier<PIPE_V>();
        Horner(result, y, 0.0360768F, count);
        Horner(result, y, 0.2659732F, count);
        Horner(result, y, 1.2067492F, count);
        Horner(result, y, 3.0899424F, count);
        Horner(result, y, 3.5156229F, count);
        Horner(result, y, 1.0F, count);
        Muls(temporary, x, -1.0F, count);
        PipeBarrier<PIPE_V>();
        Exp(temporary, temporary, count);
        PipeBarrier<PIPE_V>();
        Mul(result, result, temporary, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ static inline void LargeI0e(LocalTensor<float> x, LocalTensor<float> y, LocalTensor<float> temporary,
                                           LocalTensor<float> result, uint32_t count)
    {
        Maxs(y, x, 3.75F, count);
        PipeBarrier<PIPE_V>();
        Sqrt(temporary, y, count);
        PipeBarrier<PIPE_V>();
        Reciprocal(y, y, count);
        PipeBarrier<PIPE_V>();
        Muls(y, y, 3.75F, count);
        Duplicate(result, 3.92377e-3F, count);
        PipeBarrier<PIPE_V>();
        Horner(result, y, -0.01647633F, count);
        Horner(result, y, 0.02635537F, count);
        Horner(result, y, -0.02057706F, count);
        Horner(result, y, 9.16281e-3F, count);
        Horner(result, y, -1.57565e-3F, count);
        Horner(result, y, 2.25319e-3F, count);
        Horner(result, y, 0.01328592F, count);
        Horner(result, y, 0.39894228F, count);
        Div(result, result, temporary, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ static inline void Run(LocalTensor<float> scratch, LocalTensor<uint8_t> mask, uint32_t offset,
                                      uint32_t count, const AsdSip::KaiserWindowTilingData& data)
    {
        auto temporary = scratch;
        auto x = scratch[TILE_ELEMENTS];
        auto exponent = scratch[2 * TILE_ELEMENTS];
        auto y = scratch[3 * TILE_ELEMENTS];
        auto small = scratch[4 * TILE_ELEMENTS];
        auto large = scratch[5 * TILE_ELEMENTS];
        auto indices = temporary.ReinterpretCast<int32_t>();
        const int32_t first = static_cast<int32_t>(static_cast<int64_t>(offset) * 2 - data.denominator);
        MakeNumerator(indices, first, count);
        PipeBarrier<PIPE_V>();
        Cast(x, indices, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Muls(x, x, data.inverseN, count);
        PipeBarrier<PIPE_V>();
        Mul(x, x, x, count);
        PipeBarrier<PIPE_V>();
        Muls(x, x, -1.0F, count);
        PipeBarrier<PIPE_V>();
        Adds(x, x, 1.0F, count);
        PipeBarrier<PIPE_V>();
        Abs(x, x, count);
        PipeBarrier<PIPE_V>();
        Sqrt(x, x, count);
        PipeBarrier<PIPE_V>();
        // Retain beta*(t-1), instead of subtracting two large rounded products.
        Adds(exponent, x, -1.0F, count);
        Muls(x, x, data.beta, count);
        PipeBarrier<PIPE_V>();
        Muls(exponent, exponent, data.beta, count);
        PipeBarrier<PIPE_V>();
        Exp(exponent, exponent, count);
        PipeBarrier<PIPE_V>();
        const int32_t last = first + static_cast<int32_t>(2 * (count - 1));
        const int32_t absFirst = first < 0 ? -first : first;
        const int32_t absLast = last < 0 ? -last : last;
        const float furthestR = static_cast<float>(absFirst > absLast ? absFirst : absLast) * data.inverseN;
        const float minimumXSquared = data.beta * data.beta * (1.0F - furthestR * furthestR);
        if (data.beta <= 3.75F) {
            SmallI0e(x, y, temporary, small, count);
        } else if (minimumXSquared > 14.125F) {
            // 14.125 > 3.75^2 leaves a conservative margin for float rounding.
            // Only tiles safely inside the large-x interval take this branch.
            LargeI0e(x, y, temporary, small, count);
        } else {
            SmallI0e(x, y, temporary, small, count);
            LargeI0e(x, y, temporary, large, count);
            // CompareScalar on arch22 emits complete 64-float repeats only.
            // Select and the GM copy still consume only the valid tail count.
            const uint32_t compareCount = (count + 63U) / 64U * 64U;
            CompareScalar(mask, x, 3.75F, CMPMODE::LT, compareCount);
            PipeBarrier<PIPE_V>();
            Select(small, mask, small, large, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }
        PipeBarrier<PIPE_V>();
        Mul(small, small, exponent, count);
        PipeBarrier<PIPE_V>();
        Muls(small, small, data.inverseI0e, count);
    }
};
} // namespace KaiserWindow
#endif
