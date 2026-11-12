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
 * @brief 复数矩阵乘加算子 (Batch CGEMM): C = alpha * op(A) * op(B) + beta * C
 */
at::Tensor asdBlasCgemmBatched(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, const at::Scalar& alpha,
                               const at::Scalar& beta, int64_t transA, int64_t transB)
{
    // 1. 基本校验：支持 ComplexFloat (FP32) 和 ComplexHalf (FP16)
    TORCH_CHECK((A.scalar_type() == at::kComplexFloat || A.scalar_type() == at::kComplexHalf) &&
                    A.scalar_type() == B.scalar_type() && A.scalar_type() == C.scalar_type(),
                "asd_blas_cgemm_batched: Tensors must be ComplexFloat or ComplexHalf and have "
                "matching types.");
    TORCH_CHECK(A.dim() == sip_pta::DIM_3 && B.dim() == sip_pta::DIM_3 && C.dim() == sip_pta::DIM_3,
                "asd_blas_cgemm_batched: Input tensors must be 3D [Batch, M, N].");

    // 2. 数值与转置校验
    c10::complex<double> alpha_val = alpha.to<c10::complex<double>>();
    TORCH_CHECK(alpha_val.real() == 1.0 && alpha_val.imag() == 0.0, "asd_blas_cgemm_batched: alpha must be 1.0.");
    c10::complex<double> beta_val = beta.to<c10::complex<double>>();
    TORCH_CHECK(beta_val.real() == 0.0 && beta_val.imag() == 0.0, "asd_blas_cgemm_batched: beta must be 0.0.");

    TORCH_CHECK(transA == 0, "asd_blas_cgemm_batched: transA must be 0 (No Transpose).");
    TORCH_CHECK(transB == 0, "asd_blas_cgemm_batched: transB must be 0 (No Transpose).");

    // 形状一致性校验（issue #386）：batchCount/m/n/k 均从单侧张量提取，A/B/C 批数
    // 不一致时按 C 的 batchCount 下传存在 A/B 越界读风险；k/m/n 须按 transA/transB==0
    // 语义匹配（m=A.size(1)=C.size(1)，k=A.size(2)=B.size(1)，n=B.size(2)=C.size(2)）
    TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == C.size(0),
                "asd_blas_cgemm_batched: Batch counts of A/B/C must match, got A=", A.size(0), " B=", B.size(0),
                " C=", C.size(0));
    TORCH_CHECK(A.size(1) == C.size(1) && A.size(2) == B.size(1) && B.size(2) == C.size(2),
                "asd_blas_cgemm_batched: A/B/C shapes do not match under transA=transB=0: expected "
                "m=A.size(1)=C.size(1), k=A.size(2)=B.size(1), n=B.size(2)=C.size(2), got A [",
                A.size(1), ", ", A.size(2), "], B [", B.size(1), ", ", B.size(2), "], C [", C.size(1), ", ", C.size(2),
                "]");

    c10_npu::NPUGuard guard(A.device());

    // 3. 参数准备
    AsdSip::asdBlasOperation_t opA = static_cast<AsdSip::asdBlasOperation_t>(transA);
    AsdSip::asdBlasOperation_t opB = static_cast<AsdSip::asdBlasOperation_t>(transB);

    int64_t batchCount = C.size(0);
    int64_t m = C.size(1);
    int64_t n = C.size(2);
    int64_t k = (opA == AsdSip::asdBlasOperation_t::ASDBLAS_OP_N) ? A.size(2) : A.size(1);

    int64_t lda = A.size(2);
    int64_t ldb = B.size(2);
    int64_t ldc = C.size(2);

    aclTensor* acl_a = CreateAclTensorFromAtTensor(A);
    aclTensor* acl_b = CreateAclTensorFromAtTensor(B);
    aclTensor* acl_c = CreateAclTensorFromAtTensor(C);

    // 4. 根据数据类型分发调用
    if (A.scalar_type() == at::kComplexFloat) {
        // 单精度逻辑 (Complex64)
        std::complex<float> a_std(1.0f, 0.0f);
        std::complex<float> b_std(0.0f, 0.0f);

        EXEC_BLAS_FUNC(AsdSip::asdBlasCgemmBatched, AsdSip::asdBlasMakeCgemmBatchedPlan, {}, opA, opB, m, n, k, a_std,
                       acl_a, lda, acl_b, ldb, b_std, acl_c, ldc, batchCount);
    } else {
        // 半精度逻辑 (Complex32)
        std::complex<op::fp16_t> a_std(op::fp16_t(1.0f), op::fp16_t(0.0f));
        std::complex<op::fp16_t> b_std(op::fp16_t(0.0f), op::fp16_t(0.0f));

        EXEC_BLAS_FUNC(AsdSip::asdBlasHCgemmBatched, AsdSip::asdBlasMakeHCgemmBatchedPlan, {}, opA, opB, m, n, k, a_std,
                       acl_a, lda, acl_b, ldb, b_std, acl_c, ldc, batchCount);
    }

    return C;
}
// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_cgemm_batched(Tensor A, Tensor B, Tensor(a!) C, Scalar alpha, Scalar beta, int "
          "transA, int transB) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_cgemm_batched", &asdBlasCgemmBatched); }
} // namespace sip_pta
