/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_KAISER_WINDOW_H
#define ASDSIP_PARAMS_KAISER_WINDOW_H
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace AsdSip {
namespace OpParam {
struct KaiserWindow {
    int64_t windowLength = 0;
    bool periodic = true;
    double beta = 12.0;
    int32_t dtype = 0;

    bool operator==(const KaiserWindow& other) const
    {
        return windowLength == other.windowLength && periodic == other.periodic && dtype == other.dtype &&
               (std::memcmp(&beta, &other.beta, sizeof(beta)) == 0 || (std::isnan(beta) && std::isnan(other.beta)));
    }

    std::string ToString() const
    {
        std::ostringstream os;
        os << "KaiserWindow: length=" << windowLength << ", periodic=" << periodic << ", beta=" << beta
           << ", dtype=" << dtype;
        return os.str();
    }
};
} // namespace OpParam
} // namespace AsdSip
#endif
