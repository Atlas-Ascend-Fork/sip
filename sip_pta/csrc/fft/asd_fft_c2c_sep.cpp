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
#include <tuple>
#include <vector>

#include "pytorch_npu_helper.hpp"

namespace sip_pta {
// -----------------------------------------------------------------------------------
// 实部虚部分离版 C2C FFT (支持 1D/2D/3D)
// -----------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor> asdFftC2CSep(const at::Tensor& in_real, const at::Tensor& in_imag, bool isForward)
{
    c10_npu::NPUGuard guard(in_real.device());

    // dtype 校验与同族入口口径对齐（issue #371）：分离实虚部契约为 float32，
    // 错误 dtype 不再下传 plan 与内核调用链
    TORCH_CHECK(in_real.scalar_type() == at::kFloat && in_imag.scalar_type() == at::kFloat,
                "asd_fft_c2c_sep: input real/imag must be float32, got ", in_real.scalar_type(), "/",
                in_imag.scalar_type());

    // 1. 连续性检查
    at::Tensor input_real = in_real.is_contiguous() ? in_real : in_real.contiguous();
    at::Tensor input_imag = in_imag.is_contiguous() ? in_imag : in_imag.contiguous();

    // 2. 准备输出
    at::Tensor output_real = at::empty_like(input_real);
    at::Tensor output_imag = at::empty_like(input_imag);

    auto selfShape = input_real.sizes();
    int64_t dim = selfShape.size();

    // 3. 维度校验2 或 4
    TORCH_CHECK(dim == 2 || dim == 4, "Separated FFT support 2 or 4 dimensions (including Batch), but got ", dim);
    TORCH_CHECK(input_real.sizes() == input_imag.sizes(), "Real and Imaginary tensors must have the same shape");

    op_api::FFTParam param;

    if (isForward) {
        param.direction = AsdSip::ASCEND_FFT_FORWARD;
    } else {
        param.direction = AsdSip::ASCEND_FFT_INVERSE;
    }

    param.batchSize = selfShape[0];
    param.fftType = AsdSip::ASCEND_FFT_C2C_SEP;
    param.dimType = AsdSip::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
    // 5. 分支逻辑
    if (dim == sip_pta::DIM_2) {
        // --- 1D FFT: [Batch, N] ---
        param.fftXSize = selfShape[1];
    } else if (dim == sip_pta::DIM_4) {
        // --- 3D FFT: [Batch, D, H, W] ---
        param.fftXSize = selfShape[sip_pta::DIM_1];
        param.fftYSize = selfShape[sip_pta::DIM_2];
        param.fftZSize = selfShape[sip_pta::DIM_3];
    }
    EXEC_FFT_FUNC(AsdSip::asdFftExecC2CSeparated, param, input_real, input_imag, output_real, output_imag);
    return {output_real, output_imag};
}

// -----------------------------------------------------------------------------------
// 算子注册逻辑
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    // 定义接口：输入两个 Tensor，返回两个 Tensor 的列表
    m.def("asd_fft_c2c_sep(Tensor in_real, Tensor in_imag, bool is_forward) -> (Tensor, Tensor)", &asdFftC2CSep);
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_fft_c2c_sep", &asdFftC2CSep); }
} // namespace sip_pta
