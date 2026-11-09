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
#include "params/fft_mixed_common.h"
namespace AsdSip::fft1d {
inline bool CommonMixedEligible(int64_t n, int64_t batch, int pass, bool inverse, bool ascend950, bool horizontal)
{
    if (!ascend950 || !horizontal || pass < 0 || pass > 2)
        return false;
    // The established real selector uses DFT for all non-power-of-two N <= 1024.
    if (pass != 0 && n <= 1024)
        return false;
    return OpParam::ValidMixedCommon({n, batch, static_cast<OpParam::MixedCommonPass>(pass), inverse});
}
} // namespace AsdSip::fft1d
