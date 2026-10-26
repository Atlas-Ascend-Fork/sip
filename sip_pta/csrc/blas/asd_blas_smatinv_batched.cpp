/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
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
 * @brief 批量实数矩阵求逆算子 (SmatinvBatched), 支持 Float32 / Float16
 */
at::Tensor asdBlasSmatinvBatched(const at::Tensor& A)
{
    TORCH_CHECK((A.scalar_type() == at::kFloat || A.scalar_type() == at::kHalf),
                "asd_blas_smatinv_batched: Tensor must be Float32 or Float16.");

    TORCH_CHECK(A.dim() == sip_pta::DIM_3 && A.size(1) == A.size(sip_pta::DIM_2),
                "asd_blas_smatinv_batched: Tensor must be a 3D square matrix [Batch, N, N].");

    c10_npu::NPUGuard guard(A.device());
    int64_t batchSize = A.size(0);
    int64_t n = A.size(1);

    // 依据算子定义：lda 和 lda_inv 仅支持等于 n
    int64_t lda = n;
    int64_t lda_inv = n;

    at::Tensor Ainv = at::empty_like(A);
    at::Tensor info; // 实际不生效

    std::vector<int64_t> params = {n, batchSize};
    aclTensor* acl_a = CreateAclTensorFromAtTensor(A);
    aclTensor* acl_a_inv = CreateAclTensorFromAtTensor(Ainv);

    if (A.scalar_type() == at::kFloat) {
        // 单精度逻辑 (Float32)
        auto makePlan = [n, batchSize](AsdSip::asdBlasHandle handle) {
            return AsdSip::asdBlasMakeSmatinvBatchedPlan(handle, n, batchSize);
        };

        EXEC_BLAS_FUNC(AsdSip::asdBlasSmatinvBatched, makePlan, params, n, acl_a, lda, acl_a_inv, lda_inv, info,
                       batchSize);
    } else {
        // 半精度逻辑 (Float16, kernel 内部以 FP32 计算后回转)
        auto makePlan = [n, batchSize](AsdSip::asdBlasHandle handle) {
            return AsdSip::asdBlasMakeSmatinvBatchedPlan(handle, n, batchSize);
        };

        EXEC_BLAS_FUNC(AsdSip::asdBlasHSmatinvBatched, makePlan, params, n, acl_a, lda, acl_a_inv, lda_inv, info,
                       batchSize);
    }

    return Ainv;
}

// 算子注册
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_blas_smatinv_batched(Tensor A) -> Tensor"); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_smatinv_batched", &asdBlasSmatinvBatched); }
} // namespace sip_pta
