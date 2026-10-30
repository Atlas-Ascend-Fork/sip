/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_IFFTSHIFT_TILING_DATA
#define ASDOPS_IFFTSHIFT_TILING_DATA

#include <cstdint>

namespace AsdSip {

struct IfftShiftTilingData {
    int64_t row{0};
    int64_t col{0};
    int64_t batch{0};
    int64_t dist{0};
    int64_t totalRows{0};
    int64_t ubSize{0}; // per-core Unified Buffer size in bytes (platform-dependent)
    float scale{1.0f}; // element-wise output multiplier (1.0f = pure permutation)
};
} // namespace AsdSip
#endif
