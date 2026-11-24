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
#include <cstdint>

namespace AsdSip::OpParam {
// For N=32*(i+1) and batch=N, 0 selects mixed Vec; power-of-two entries are ignored.
inline constexpr uint8_t REAL_TARGET_TAILS[256] = {
    0,   0,   0,   0,  0,   0,   0,   0,  0,   0,   0,   0,  0,   0,   0,   0, // 32..512
    0,   0,   0,   0,  0,   0,   0,   0,  0,   0,   0,   0,  29,  0,   31,  0, // 544..1024
    0,   0,   0,   0,  37,  0,   0,   0,  41,  0,   43,  0,  0,   0,   47,  0, // 1056..1536
    0,   0,   0,   0,  53,  0,   55,  0,  0,   0,   59,  0,  61,  31,  0,   0, // 1568..2048
    0,   0,   67,  0,  69,  0,   71,  0,  73,  0,   0,   0,  77,  0,   79,  0, // 2080..2560
    0,   0,   83,  0,  85,  43,  87,  0,  89,  0,   91,  0,  93,  47,  95,  0, // 2592..3072
    97,  0,   99,  0,  101, 0,   103, 0,  0,   53,  107, 0,  109, 55,  111, 0, // 3104..3584
    113, 0,   115, 0,  117, 59,  119, 0,  121, 61,  123, 0,  0,   0,   127, 0, // 3616..4096
    129, 0,   131, 0,  0,   67,  0,   0,  137, 69,  139, 0,  47,  71,  0,   0, // 4128..4608
    0,   73,  0,   0,  149, 0,   151, 0,  0,   0,   0,   0,  157, 79,  53,  0, // 4640..5120
    0,   0,   163, 0,  0,   83,  167, 0,  0,   85,  0,   0,  173, 87,  0,   0, // 5152..5632
    59,  89,  179, 0,  181, 91,  61,  0,  0,   93,  0,   47, 0,   95,  191, 0, // 5664..6144
    193, 97,  0,   0,  197, 0,   199, 0,  67,  101, 0,   0,  0,   103, 0,   0, // 6176..6656
    0,   0,   211, 53, 71,  107, 215, 0,  0,   109, 73,  0,  0,   111, 223, 0, // 6688..7168
    0,   113, 227, 0,  229, 115, 0,   29, 233, 0,   235, 59, 79,  119, 239, 0, // 7200..7680
    241, 121, 0,   61, 0,   123, 247, 31, 83,  0,   251, 0,  253, 127, 85,  0, // 7712..8192
};
} // namespace AsdSip::OpParam
