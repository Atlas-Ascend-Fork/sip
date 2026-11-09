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
 * @brief 复数矩阵按行缩放算子 (ColwiseMul): B(i,j) = A(i,j) * v(i)
 */
at::Tensor asdBlasColwiseMul(const at::Tensor& mat, const at::Tensor& vec)
{
    // 基本校验：确保输入为复数浮点类型
    TORCH_CHECK(mat.scalar_type() == at::kComplexFloat && vec.scalar_type() == at::kComplexFloat,
                "asd_blas_colwise_mul: Only ComplexFloat (Complex64) tensors are supported.");
    // 维度一致性校验（issue #372，对齐 strmv/ctrmv 口径）：mat 必须为 2D、vec 为 1D
    // 且长度与 mat 行数一致；原实现对 1D mat 直接取 size(1) 抛底层 IndexError、
    // 2D vec 也能通过 vec.size(0)==m 的宽松校验
    TORCH_CHECK(mat.dim() == 2 && vec.dim() == 1,
                "asd_blas_colwise_mul: mat must be 2D and vec must be 1D, got mat.dim()=", mat.dim(),
                " vec.dim()=", vec.dim());

    c10_npu::NPUGuard guard(mat.device());
    // 解析维度：m 为行数，n 为列数
    int64_t m = mat.size(0);
    int64_t n = mat.size(1);

    // 校验向量长度是否与矩阵行数匹配
    TORCH_CHECK(vec.size(0) == m, "asd_blas_colwise_mul: vector size must match matrix row count (m).");

    // 准备输出 Tensor：与输入矩阵 mat 形状相同
    at::Tensor result = at::empty_like(mat);

    // 执行算子调用
    // 接口顺序: (handle, m, n, mat, vec, result)
    EXEC_BLAS_FUNC(AsdSip::asdBlasColwiseMul, asdBlasMakeColwiseMulPlan, {}, m, n, mat, vec, result);

    return result;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_blas_colwise_mul(Tensor mat, Tensor vec) -> Tensor"); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_colwise_mul", &asdBlasColwiseMul); }
} // namespace sip_pta
