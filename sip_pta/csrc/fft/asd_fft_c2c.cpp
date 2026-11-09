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

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace sip_pta {
at::Tensor asdFftC2C(const at::Tensor& in, bool isForward)
{
    c10_npu::NPUGuard guard(in.device());

    // dtype 校验与同族 asd_fft_shift 口径对齐（issue #371）：C2C 契约为 complex64，
    // 错误 dtype 不再下传 plan 与内核调用链
    TORCH_CHECK(in.scalar_type() == at::kComplexFloat, "asd_fft_c2c: input must be complex64, got ", in.scalar_type());
    at::Tensor input = in.is_contiguous() ? in : in.contiguous();
    at::Tensor output = at::empty_like(input);
    auto selfShape = input.sizes();
    TORCH_CHECK(selfShape.size() >= sip_pta::DIM_2 && selfShape.size() <= sip_pta::DIM_4,
                "Input tensor must have 2, 3 or 4 dimensions, but got ", selfShape.size());
    op_api::FFTParam param;

    if (isForward) {
        param.direction = AsdSip::ASCEND_FFT_FORWARD;
    } else {
        param.direction = AsdSip::ASCEND_FFT_INVERSE;
    }
    param.batchSize = selfShape[0];
    param.fftType = AsdSip::ASCEND_FFT_C2C;
    param.dimType = AsdSip::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
    if (selfShape.size() == sip_pta::DIM_2) {
        // 1D
        param.fftXSize = selfShape[sip_pta::DIM_1];
    } else if (selfShape.size() == sip_pta::DIM_3) {
        // 2D
        param.fftXSize = selfShape[sip_pta::DIM_1];
        param.fftYSize = selfShape[sip_pta::DIM_2];
    } else if (selfShape.size() == sip_pta::DIM_4) {
        // 3D
        param.fftXSize = selfShape[sip_pta::DIM_1];
        param.fftYSize = selfShape[sip_pta::DIM_2];
        param.fftZSize = selfShape[sip_pta::DIM_3];
    }
    EXEC_FFT_FUNC(AsdSip::asdFftExecC2C, param, input, output);
    return output;
}

// 2. 算子注册 (Fragment)
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_fft_c2c(Tensor input, bool is_forward) -> Tensor", &asdFftC2C); }
TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_fft_c2c", &asdFftC2C); }
} // namespace sip_pta
