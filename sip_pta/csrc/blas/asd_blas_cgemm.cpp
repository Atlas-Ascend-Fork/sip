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

#include "blas_api.h"
#include "pytorch_npu_helper.hpp"

namespace sip_pta {
/**
 * @brief 复数矩阵乘加算子 (CGEMM): C = alpha * op(A) * op(B) + beta * C
 * A, B, C: ComplexFloat (Complex64)
 * transA, transB: int (0: N, 1: T, 2: C)
 */
at::Tensor asdBlasCgemm(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, const at::Scalar& alpha,
                        const at::Scalar& beta, int64_t transA, int64_t transB)
{
    // 基本校验：必须是复数类型
    TORCH_CHECK(A.scalar_type() == at::kComplexFloat && B.scalar_type() == at::kComplexFloat &&
                    C.scalar_type() == at::kComplexFloat,
                "asd_blas_cgemm: Only ComplexFloat (Complex64) tensors are supported.");

    // 枚举域与维度校验（issue #386，对齐 cgemv/#372 口径）：transA/transB 仅允许
    // {0,1,2}，A/B/C 必须为 2 维；非法枚举经 static_cast 后 opA/opB 比较结果不确定，
    // m/n/k/lda 推导随之错乱，不再静默下传内核
    TORCH_CHECK(transA == 0 || transA == 1 || transA == 2, "asd_blas_cgemm: transA must be 0 (N), 1 (T) or 2 (C), got ",
                transA);
    TORCH_CHECK(transB == 0 || transB == 1 || transB == 2, "asd_blas_cgemm: transB must be 0 (N), 1 (T) or 2 (C), got ",
                transB);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "asd_blas_cgemm: A/B/C must be 2D, got A.dim()=", A.dim(),
                " B.dim()=", B.dim(), " C.dim()=", C.dim());

    c10_npu::NPUGuard guard(A.device());

    // 处理转置枚举转换
    // 根据 enum class asdBlasOperation_t { ASDBLAS_OP_N = 0, ASDBLAS_OP_T, ASDBLAS_OP_C };
    AsdSip::asdBlasOperation_t opA = static_cast<AsdSip::asdBlasOperation_t>(transA);
    AsdSip::asdBlasOperation_t opB = static_cast<AsdSip::asdBlasOperation_t>(transB);

    // 解析维度 M, N, K
    // 矩阵 C 的维度永远是 (m, n)
    int64_t m = C.size(0);
    int64_t n = C.size(1);

    int64_t k = (opA == AsdSip::asdBlasOperation_t::ASDBLAS_OP_N) ? A.size(1) : A.size(0);

    // 形状一致性校验（issue #386）：m/n 取自 C、k 由 A 按 trans 推导，B 的 k/n 对应维
    // 须与 transB 语义一致，B/C 不匹配时按错误 ldb/k 下传内核，结果静默错误
    TORCH_CHECK((opA == AsdSip::asdBlasOperation_t::ASDBLAS_OP_N) ? (m == A.size(0) && k == A.size(1)) :
                                                                    (m == A.size(1) && k == A.size(0)),
                "asd_blas_cgemm: A shape [", A.size(0), ", ", A.size(1), "] does not match transA=", transA,
                " with C [", m, ", ", n, "]");
    TORCH_CHECK((opB == AsdSip::asdBlasOperation_t::ASDBLAS_OP_N) ? (k == B.size(0) && n == B.size(1)) :
                                                                    (k == B.size(1) && n == B.size(0)),
                "asd_blas_cgemm: B shape [", B.size(0), ", ", B.size(1), "] does not match transB=", transB,
                " with k=", k, " and C [", m, ", ", n, "]");

    // 设置 Leading Dimensions (主维步长)
    int64_t lda = A.size(0);
    int64_t ldb = B.size(0);
    int64_t ldc = C.size(0);

    // 转换标量 alpha, beta
    c10::complex<float> a_c10 = alpha.to<c10::complex<float>>();
    std::complex<float> a_std(a_c10.real(), a_c10.imag());

    c10::complex<float> b_c10 = beta.to<c10::complex<float>>();
    std::complex<float> b_std(b_c10.real(), b_c10.imag());

    // 使用宏执行算子
    auto makePlan = [opA, opB, m, n, k, lda, ldb, ldc](AsdSip::asdBlasHandle handle) {
        return AsdSip::asdBlasMakeCgemmPlan(handle, opA, opB, m, n, k, lda, ldb, ldc);
    };

    std::vector<int64_t> params = {static_cast<int64_t>(opA), static_cast<int64_t>(opB), m, n, k, lda, ldb, ldc};
    EXEC_BLAS_FUNC(AsdSip::asdBlasCgemm, makePlan, params, opA, opB, m, n, k, a_std, A, lda, B, ldb, b_std, C, ldc);

    return C;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_cgemm(Tensor A, Tensor B, Tensor(a!) C, Scalar alpha, Scalar beta, int transA, "
          "int transB) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_cgemm", &asdBlasCgemm); }
} // namespace sip_pta
