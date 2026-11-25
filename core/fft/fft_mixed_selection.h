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
#include "params/fft_c2c_dispatch.h"
#include "params/fft_mixed_common.h"
#include "params/fft_real_dispatch.h"
namespace AsdSip::fft1d {
enum class Real2dAxisRoute { Native, RegBase, Mixed };
// Ranges apply only to axes that native dispatch would send to DFT.
// Odd real rows use the full N-point transform; even rows use the packed N/2 transform.
inline bool PreferFftToRealDft(int n, int64_t rows)
{
    if (n < 16 || n > 1024 || rows < 32 || rows > 512)
        return false;
    if (n % 2 != 0)
        return n < 48 && rows <= 128 && OpParam::LargestPrimeFactor(n) <= 7;
    if (n < 128)
        return OpParam::LargestPrimeFactor(n) <= 5;
    return rows >= 128 && OpParam::LargestPrimeFactor(n) <= 7;
}
inline Real2dAxisRoute SelectReal2dAxis(int n, int64_t rows, OpParam::MixedCommonPass pass, bool inverse,
                                        bool square = false)
{
    using Pass = OpParam::MixedCommonPass;
    if (n < 1 || n > 8192 || rows < 1 || rows > 0x7fffffff / (2 * n))
        return Real2dAxisRoute::Native;
    const bool power2 = (n & (n - 1)) == 0;
    if (pass == Pass::Complex) {
        const bool shortFft = power2 && n >= 16 && n <= 64 && rows >= 32 && rows <= 4096;
        if (fft::a5::Select(n, rows) == fft::a5::Route::Dft && !shortFft && !(square && n >= 32))
            return Real2dAxisRoute::Native;
        return power2 && n >= 32 ? Real2dAxisRoute::RegBase : Real2dAxisRoute::Mixed;
    }
    const bool standard = (pass == Pass::Hermitian) == inverse;
    const auto native = fft::a5::SelectReal(n, rows, pass == Pass::Hermitian);
    // Preserve already optimized square real passes. Qualify newly reached DFT axes separately.
    const bool existingSquareFft = square && standard && OpParam::RealMixedLength(n);
    if (native == fft::a5::RealRoute::Dft && !existingSquareFft && !PreferFftToRealDft(n, rows))
        return Real2dAxisRoute::Native;
    if (power2 && n >= 32 && rows <= 8192 && standard)
        return Real2dAxisRoute::RegBase;
    return n >= 96 || native == fft::a5::RealRoute::Dft ? Real2dAxisRoute::Mixed : Real2dAxisRoute::Native;
}
inline bool CommonMixedEligible(int64_t n, int64_t batch, int pass, bool inverse, bool ascend950, bool horizontal,
                                bool cubeTailAllowed = false)
{
    if (!ascend950 || !horizontal || pass < 0 || pass > 2)
        return false;
    const OpParam::FftMixedCommon p{n,       batch, static_cast<OpParam::MixedCommonPass>(pass),
                                    inverse, false, cubeTailAllowed};
    if (pass != 0 && n <= 1024 && !cubeTailAllowed)
        return false;
    return OpParam::ValidMixedCommon(p);
}
} // namespace AsdSip::fft1d
