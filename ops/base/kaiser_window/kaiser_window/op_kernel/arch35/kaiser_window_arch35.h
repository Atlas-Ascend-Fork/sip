/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_KAISER_WINDOW_ARCH35_H
#define ASDSIP_KAISER_WINDOW_ARCH35_H
#include "../kernel_common.h"

namespace KaiserWindow {
namespace Reg = AscendC::Reg;
constexpr Reg::CastTrait INDEX_CAST_TRAIT = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__simd_callee__ inline void Horner(Reg::RegTensor<float>& value, Reg::RegTensor<float>& variable, float coefficient,
                                   Reg::MaskReg& mask)
{
    Reg::Mul(value, value, variable, mask);
    Reg::Adds(value, value, coefficient, mask);
}

// The A5 numerical loop keeps the polynomial and intermediate values in
// registers. Only the final float32 values are written back to UB for casting.
__simd_vf__ inline void ComputeArch35(__ubuf__ float* output, uint32_t offset, uint32_t count, int32_t denominator,
                                      float inverseN, float beta, float inverseI0e)
{
    constexpr uint32_t LANES = 64;
    for (uint16_t chunk = 0; chunk < (count + LANES - 1) / LANES; ++chunk) {
        uint32_t remaining = count - static_cast<uint32_t>(chunk) * LANES;
        uint32_t active = remaining < LANES ? remaining : LANES;
        Reg::MaskReg mask = Reg::UpdateMask<float>(active);
        Reg::RegTensor<int32_t> index;
        Reg::RegTensor<float> x, exponent, y, temporary, small, large, constant;
        const int32_t first = static_cast<int32_t>(
            (static_cast<int64_t>(offset) + static_cast<int64_t>(chunk) * LANES) * 2 - denominator);
        // Build the numerator in integer arithmetic before the float conversion.
        Reg::Arange(index, int32_t{0});
        Reg::Muls(index, index, int32_t{2}, mask);
        Reg::Adds(index, index, first, mask);
        Reg::Cast<float, int32_t, INDEX_CAST_TRAIT>(x, index, mask);
        Reg::Muls(x, x, inverseN, mask);
        Reg::Mul(x, x, x, mask);
        Reg::Muls(x, x, -1.0F, mask);
        Reg::Adds(x, x, 1.0F, mask);
        Reg::Abs(x, x, mask);
        Reg::Sqrt(x, x, mask);
        Reg::Adds(exponent, x, -1.0F, mask);
        Reg::Muls(exponent, exponent, beta, mask);
        Reg::Exp(exponent, exponent, mask);
        Reg::Muls(x, x, beta, mask);
        Reg::Mins(y, x, 3.75F, mask);
        Reg::Muls(y, y, 1.0F / 3.75F, mask);
        Reg::Mul(y, y, y, mask);
        Reg::Duplicate(small, 0.0045813F, mask);
        Horner(small, y, 0.0360768F, mask);
        Horner(small, y, 0.2659732F, mask);
        Horner(small, y, 1.2067492F, mask);
        Horner(small, y, 3.0899424F, mask);
        Horner(small, y, 3.5156229F, mask);
        Horner(small, y, 1.0F, mask);
        Reg::Muls(temporary, x, -1.0F, mask);
        Reg::Exp(temporary, temporary, mask);
        Reg::Mul(small, small, temporary, mask);
        Reg::Maxs(y, x, 3.75F, mask);
        Reg::Sqrt(temporary, y, mask);
        Reg::Duplicate(constant, 3.75F, mask);
        Reg::Div(y, constant, y, mask);
        Reg::Duplicate(large, 3.92377e-3F, mask);
        Horner(large, y, -0.01647633F, mask);
        Horner(large, y, 0.02635537F, mask);
        Horner(large, y, -0.02057706F, mask);
        Horner(large, y, 9.16281e-3F, mask);
        Horner(large, y, -1.57565e-3F, mask);
        Horner(large, y, 2.25319e-3F, mask);
        Horner(large, y, 0.01328592F, mask);
        Horner(large, y, 0.39894228F, mask);
        Reg::Div(large, large, temporary, mask);
        Reg::MaskReg smallMask;
        Reg::Compares<float, CMPMODE::LT>(smallMask, x, 3.75F, mask);
        Reg::Select(small, small, large, smallMask);
        Reg::Mul(small, small, exponent, mask);
        Reg::Muls(small, small, inverseI0e, mask);
        Reg::StoreAlign(output + static_cast<uint32_t>(chunk) * LANES, small, mask);
    }
}

struct Arch35Compute {
    __aicore__ static inline void Run(LocalTensor<float> scratch, LocalTensor<uint8_t>, uint32_t offset, uint32_t count,
                                      const AsdSip::KaiserWindowTilingData& data)
    {
        auto result = scratch[4 * TILE_ELEMENTS];
        ComputeArch35(reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr()), offset, count,
                      static_cast<int32_t>(data.denominator), data.inverseN, data.beta, data.inverseI0e);
    }
};
} // namespace KaiserWindow
#endif
