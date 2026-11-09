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
    std::cout << "HOST_PASS positive_grid=" << count << " negative=11\n";
}
