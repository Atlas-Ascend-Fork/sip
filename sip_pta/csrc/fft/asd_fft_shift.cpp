/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "pytorch_npu_helper.hpp"

#include "fft_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>
#include <tuple>

namespace sip_pta {

/*
 * Torch API for asdFftShift.
 *
 * Shifts the zero-frequency component of batched complex slices to the
 * center of each slice (aligned with the reference CUDA fftShiftBatch2DKernel):
 * NOTE: the current version supports 2D slices (row x col) only; support for
 * other dimensions will be added in a later release.
 *   output[s][(r + (row + 1) / 2) % row][(c + (col + 1) / 2) % col] = input[s][r][c]
 *
 * Python interface:
 *   fft_shift(input, row, col, batch, dist=0) -> Tensor
 *   fft_shift(input_3d) -> Tensor  (auto-infer from [batch, row, col])
 */
at::Tensor asdFftShift(const at::Tensor& input, int64_t row, int64_t col, int64_t batch, int64_t dist)
{
    c10_npu::NPUGuard guard(input.device());

    TORCH_CHECK(input.scalar_type() == at::kComplexFloat, "asd_fft_shift: input must be complex64, got ",
                input.scalar_type());
    TORCH_CHECK(row > 0 && col > 0 && batch > 0, "asd_fft_shift: row/col/batch must be positive, got row=", row,
                " col=", col, " batch=", batch);
    int64_t distVal = (dist > 0) ? dist : row * col;
    TORCH_CHECK(distVal >= row * col, "asd_fft_shift: dist (", distVal, ") must be >= row * col (", row * col, ")");
    int64_t requiredInNum = (batch - 1) * distVal + row * col;
    TORCH_CHECK(input.numel() >= requiredInNum, "asd_fft_shift: input needs at least ", requiredInNum,
                " elements, got ", input.numel());

    at::Tensor output = at::empty({batch, row, col}, input.options());

    auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);
    aclTensor* acl_in = CreateAclTensorFromAtTensor(input);
    aclTensor* acl_out = CreateAclTensorFromAtTensor(output);
    void* workspace_addr = nullptr;
    EXEC_FUNC(AsdSip::asdFftShift, row, col, batch, distVal, acl_in, acl_out, sip_stream, workspace_addr);
    return output;
}

// Convenience overload: infer row/col/batch from a [batch, row, col] tensor.
at::Tensor asdFftShiftAuto(const at::Tensor& input)
{
    auto inShape = input.sizes();
    TORCH_CHECK(inShape.size() == sip_pta::DIM_3,
                "asd_fft_shift: auto mode expects a 3D [batch, row, col] tensor, got ", inShape.size(), " dimensions");
    return asdFftShift(input, inShape[sip_pta::DIM_1], inShape[sip_pta::DIM_2], inShape[0], 0);
}

TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_fft_shift(Tensor input, int row, int col, int batch, int dist=0) -> Tensor");
    m.def("asd_fft_shift_auto(Tensor input) -> Tensor");
}
TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
{
    m.impl("asd_fft_shift", &asdFftShift);
    m.impl("asd_fft_shift_auto", &asdFftShiftAuto);
}
} // namespace sip_pta
