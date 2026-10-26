/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_CGETRI_BATCHED_H
#define ASDSIP_PARAMS_CGETRI_BATCHED_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {
struct SgetriBatched {
    int64_t dtype;
    int64_t n;
    int64_t batchSize;

    bool operator==(const SgetriBatched& other) const
    {
        return this->dtype == other.dtype && this->n == other.n && this->batchSize == other.batchSize;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "dtype:" << dtype << "n:" << n << ", batchSize: " << batchSize;
        return ss.str();
    }
};
} // namespace OpParam
} // namespace AsdSip

#endif // ASDSIP_PARAMS_CGETRI_BATCHED_H
