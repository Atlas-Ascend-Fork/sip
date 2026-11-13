/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_KAISER_WINDOW_TILING_DATA_H
#define ASDSIP_KAISER_WINDOW_TILING_DATA_H
#include <cstdint>
namespace AsdSip {
struct KaiserWindowTilingData {
    uint32_t length;
    uint32_t denominator;
    uint32_t perCore;
    uint32_t dtype; // 0: FLOAT32, 1: FLOAT16, 2: BFLOAT16
    float beta;
    float inverseI0e;
    float inverseN;
    uint32_t mode; // 0: normal, 1: fill one, 2: fill NaN
};
} // namespace AsdSip
#endif
