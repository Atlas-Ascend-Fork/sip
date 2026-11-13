/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <mki/utils/platform/platform_info.h>
#include "params/kaiser_window.h"
#include "tiling_data.h"
#include "utils/assert.h"
#include "kaiser_window_tiling.h"

namespace AsdSip {
using namespace Mki;
namespace {
// Scaled I0 avoids overflow near beta=88. The two polynomial approximations
// have relative error below 1e-6 over the task's [0, 88] beta domain.
double ScaledI0(double x)
{
    if (x < 3.75) {
        const double y = (x / 3.75) * (x / 3.75);
        return std::exp(-x) *
               (1.0 + y * (3.5156229 +
                           y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813))))));
    }
    const double y = 3.75 / x;
    return (0.39894228 +
            y * (0.01328592 +
                 y * (2.25319e-3 +
                      y * (-1.57565e-3 +
                           y * (9.16281e-3 +
                                y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 3.92377e-3)))))))) /
           std::sqrt(x);
}
} // namespace

AspbStatus KaiserWindowTiling(const LaunchParam& launch, KernelInfo& info)
{
    auto* data = reinterpret_cast<KaiserWindowTilingData*>(info.GetTilingHostAddr());
    ASDSIP_CHECK(data != nullptr, "Missing KaiserWindow tiling memory", return ErrorType::ACL_ERROR_INVALID_PARAM);
    const auto& param = AnyCast<OpParam::KaiserWindow>(launch.GetParam());
    const uint32_t length = static_cast<uint32_t>(param.windowLength);
    const uint32_t maxCores = std::max(
        1U, static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR)));
    constexpr uint32_t MIN_CORE_ELEMENTS = 256;
    constexpr uint32_t ALIGN_ELEMENTS = 64;
    const uint32_t cores = std::min(maxCores, std::max(1U, (length + MIN_CORE_ELEMENTS - 1) / MIN_CORE_ELEMENTS));
    const uint32_t perCore = ((length + cores - 1) / cores + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;
    data->length = length;
    data->denominator = length > 1 ? length - (param.periodic ? 0 : 1) : 1;
    data->perCore = std::max(ALIGN_ELEMENTS, perCore);
    data->dtype = param.dtype == TENSOR_DTYPE_FLOAT ? 0 : (param.dtype == TENSOR_DTYPE_FLOAT16 ? 1 : 2);
    data->beta = static_cast<float>(std::abs(param.beta));
    data->mode = length == 1 || std::fpclassify(param.beta) == FP_ZERO ? 1 : (!std::isfinite(data->beta) ? 2 : 0);
    data->inverseN = 1.0F / static_cast<float>(data->denominator);
    data->inverseI0e = data->mode == 0 ? static_cast<float>(1.0 / ScaledI0(std::abs(param.beta))) : 1.0F;
    info.SetBlockDim(std::max(1U, (length + data->perCore - 1) / data->perCore));
    info.GetScratchSizes().push_back(0);
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
