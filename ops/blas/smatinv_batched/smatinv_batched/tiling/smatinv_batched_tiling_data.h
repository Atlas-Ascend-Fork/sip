/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SMATINV_BATCHED_TILING_DATA_H
#define SMATINV_BATCHED_TILING_DATA_H

#include <cstdint>

namespace AsdSip {
struct SmatinvBatchedTilingData {
    uint32_t dtype;           // 0-fp32; 1-fp16
    uint32_t n;               // matrix row size
    uint32_t startOffset[40]; // per-vector-core element offset
    uint32_t calNum[40];      // per-vector-core matrix num
};
} // namespace AsdSip

#endif // SMATINV_BATCHED_TILING_DATA_H
