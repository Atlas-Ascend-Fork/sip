/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_FFT_TRANSPOSE_H
#define ASDSIP_PARAMS_FFT_TRANSPOSE_H
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace AsdSip::OpParam {
// Contiguous complex64 batch x rows x cols -> batch x cols x rows.
struct FftTranspose {
    int64_t batch = 0;
    int64_t rows = 0;
    int64_t cols = 0;

    bool Valid() const
    {
        return rows > 0 && rows <= 32768 && cols > 0 && cols <= 32768 && batch > 0 &&
               batch <= std::numeric_limits<int32_t>::max() / (rows * cols);
    }
    bool operator==(const FftTranspose& p) const { return batch == p.batch && rows == p.rows && cols == p.cols; }
    std::string ToString() const
    {
        std::ostringstream s;
        s << "FftTranspose batch=" << batch << " rows=" << rows << " cols=" << cols;
        return s.str();
    }
};
} // namespace AsdSip::OpParam
#endif
