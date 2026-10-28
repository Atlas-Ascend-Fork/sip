/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDSIP_PARAMS_IFFTSHIFT_H
#define ASDSIP_PARAMS_IFFTSHIFT_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {

/**
 * Parameter struct for IfftShift (Ascend 950) operation.
 *
 * NOTE: the current version supports 2D slices (row x col) only; support for
 * other dimensions will be added in a later release.
 *
 * Semantics (aligned with the reference CUDA ifftShiftScaleBatch2DKernel):
 *   The input is viewed as `batch` 2D slices of `row` x `col` complex elements.
 *   Slice s starts at element offset `s * dist` in the input tensor.
 *   The output is a contiguous `batch` x `row` x `col` tensor where
 *   dst[s][(r + row / 2) % row][(c + col / 2) % col] = scale * src[s][r][c],
 *   i.e. the inverse of FftShift: for even dimensions the permutations
 *   coincide, for odd dimensions the shifts differ by one (floor instead of
 *   ceil). `scale` is an element-wise output multiplier so that the inverse
 *   FFT normalization (1 / (row * col)) can be folded into the shift pass.
 *
 * Fields:
 *   row   - Number of rows of each 2D slice (iRow)
 *   col   - Number of columns of each 2D slice (iCol)
 *   batch - Number of 2D slices (iBatch)
 *   dist  - Distance between two consecutive input slices in complex elements (iDist).
 *           Non-positive value means the input slices are contiguous (dist = row * col).
 *   scale - Element-wise output multiplier (converted to float32 in the kernel);
 *           1.0 leaves the values bit-exact unchanged (pure permutation).
 */
struct IfftShift {
    int64_t row;
    int64_t col;
    int64_t batch;
    int64_t dist;
    double scale;

    bool operator==(const IfftShift& other) const
    {
        return this->row == other.row && this->col == other.col && this->batch == other.batch &&
               this->dist == other.dist && this->scale == other.scale;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: IfftShift"
           << ", row:" << row << ", col:" << col << ", batch:" << batch << ", dist:" << dist << ", scale:" << scale;
        return ss.str();
    }
};

} // namespace OpParam
} // namespace AsdSip

#endif // ASDSIP_PARAMS_IFFTSHIFT_H
