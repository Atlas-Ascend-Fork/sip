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

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace sip_pta {
/**
 * @brief 向量乘加算子 (CAXPY): y = alpha * x + y
 * x, y: ComplexFloat
 * alpha: Complex (std::complex<float>)
 */
at::Tensor asdBlasCaxpy(const at::Tensor& x, at::Tensor& y, const at::Scalar& alpha)
{
    c10_npu::NPUGuard guard(x.device());

    at::Tensor x_fixed = x.is_contiguous() ? x : x.contiguous();
    // y 声明为原位参数 Tensor(a!)（issue #282）：对非连续 y 静默转拷贝会使
    // 调用方原 y 存储永远不被更新、原位契约静默失效，改为与 csrot/ssyr/copy
    // 等同族绑定一致的显式拒绝
    TORCH_CHECK(y.is_contiguous(), "asd_blas_caxpy: y must be contiguous.");

    int64_t n = x_fixed.numel();
    int64_t incx = 1;
    int64_t incy = 1;

    // 2. 类型与维度初步校验
    TORCH_CHECK(x.scalar_type() == at::kComplexFloat && y.scalar_type() == at::kComplexFloat,
                "asd_blas_caxpy: Only ComplexFloat (Complex64) tensors are supported.");
    TORCH_CHECK(x.numel() == y.numel(), "asd_blas_caxpy: x and y must have the same number of elements.");

    // 3. 转换 alpha 为 std::complex<float>
    c10::complex<float> a_c10 = alpha.to<c10::complex<float>>();
    std::complex<float> a_std(a_c10.real(), a_c10.imag());

    EXEC_BLAS_FUNC(AsdSip::asdBlasCaxpy, AsdSip::asdBlasMakeCaxpyPlan, {}, n, a_std, x_fixed, incx, y, incy);

    return y;
}

// -----------------------------------------------------------------------------------
// 算子注册 (Fragment)
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_blas_caxpy(Tensor x, Tensor(a!) y, Scalar alpha) -> Tensor(a!)"); }
TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_caxpy", &asdBlasCaxpy); }
} // namespace sip_pta
