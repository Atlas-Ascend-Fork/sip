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

#include "blas_api.h"

#include <ATen/ATen.h>
#include <ATen/ScalarOps.h>
#include <acl/acl.h>
#include <complex>
#include <dlfcn.h>
#include <string>

namespace sip_pta {
/**
 * @brief 向量缩放算子 (CAL)
 * sscal:  float * float
 * csscal: complex * float
 * cscal:  complex * complex
 */
/**
 * @brief 向量缩放算子 (Scal)
 */
at::Tensor asdBlasCal(const at::Tensor& x, const at::Scalar& alpha)
{
    c10_npu::NPUGuard guard(x.device());

    // 原地语义（schema 声明 Tensor(a!)，issue #283）：连续 x 直接在原存储上缩放；
    // 非连续 x 在 contiguous 拷贝上计算后 copy_ 回写原存储，保证连续/非连续
    // 两条路径的别名行为一致（输入均被改写）
    at::Tensor x_orig = x;
    at::Tensor x_fixed = x_orig.is_contiguous() ? x_orig : x_orig.contiguous();
    int64_t n = x_fixed.numel();
    int64_t incx = 1;
    auto x_type = x.scalar_type();

    // 检查 alpha 是否为复数
    bool alpha_is_complex = alpha.isComplex();

    if (x_type == at::kFloat) {
        // --- 严格校验：实数向量不能用复数标量缩放 ---
        TORCH_CHECK(!alpha_is_complex, "asd_blas_cal: Cannot scale a Float32 tensor with a complex alpha.");

        float a = alpha.to<float>();
        EXEC_BLAS_FUNC(AsdSip::asdBlasSscal, AsdSip::asdBlasMakeCalPlan, {}, n, a, x_fixed, incx);
    } else if (x_type == at::kComplexFloat) {
        if (alpha_is_complex) {
            // --- cscal: Complex * Complex ---
            c10::complex<float> a_c10 = alpha.to<c10::complex<float>>();
            std::complex<float> a(a_c10.real(), a_c10.imag());
            EXEC_BLAS_FUNC(AsdSip::asdBlasCscal, AsdSip::asdBlasMakeCalPlan, {}, n, a, x_fixed, incx);
        } else {
            // --- csscal: Complex * Float ---
            float a = alpha.to<float>();
            EXEC_BLAS_FUNC(AsdSip::asdBlasCsscal, AsdSip::asdBlasMakeCalPlan, {}, n, a, x_fixed, incx);
        }
    } else {
        TORCH_CHECK(false, "asd_blas_cal only supports Float32 or Complex64.");
    }

    if (!x_orig.is_contiguous()) {
        x_orig.copy_(x_fixed);
    }

    return x_orig;
}

// -----------------------------------------------------------------------------------
// 算子注册 (Fragment)
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_blas_cal(Tensor(a!) x, Scalar alpha) -> Tensor(a!)"); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_cal", &asdBlasCal); }
} // namespace sip_pta
