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

#include "interp_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace sip_pta {

/**
 * @brief 向量插值算子 (InterpWithCoeff)
 */
at::Tensor asdInterpWithCoeff(const at::Tensor& x, const at::Tensor& coefficient)
{
    c10_npu::NPUGuard guard(x.device());

    // 校验：数据类型必须一致，且为支持的复数类型 (ComplexFloat 或 ComplexHalf)
    TORCH_CHECK(x.scalar_type() == coefficient.scalar_type(),
                "asd_interp_with_coeff: Tensors x and coefficient must have the same dtype.");
    TORCH_CHECK(x.scalar_type() == at::kComplexFloat || x.scalar_type() == at::kComplexHalf,
                "asd_interp_with_coeff: Tensors must be ComplexFloat or ComplexHalf.");

    // 校验：维度必须为 3D (batch, m, k)
    TORCH_CHECK(x.dim() == sip_pta::DIM_3 && coefficient.dim() == sip_pta::DIM_3,
                "asd_interp_with_coeff: Inputs must be 3D tensors.");
    TORCH_CHECK(x.size(0) == coefficient.size(0), "asd_interp_with_coeff: Batch sizes must match.");
    TORCH_CHECK(x.size(sip_pta::DIM_1) == coefficient.size(sip_pta::DIM_2),
                "asd_interp_with_coeff: Inner dimensions must match (x.size(1) == coeff.size(2)).");

    // 计算输出形状: [batch, coeff.size(1), x.size(2)]
    int64_t batch = x.size(0);
    int64_t out_m = coefficient.size(1); // 14 - nRs
    int64_t out_n = x.size(2);           // totalSubcarrier

    at::Tensor output = at::empty({batch, out_m, out_n}, x.options());

    aclTensor* acl_x = CreateAclTensorFromAtTensor(x);
    aclTensor* acl_coefficient = CreateAclTensorFromAtTensor(coefficient);
    aclTensor* acl_output = CreateAclTensorFromAtTensor(output);

    // 获取 workspace 大小（检查返回值，口径对齐 #259，issue #286）
    size_t workspace_size = 0;
    TORCH_CHECK(AsdSip::asdInterpWithCoeffGetWorkspaceSize(workspace_size) == AsdSip::ErrorType::ACL_SUCCESS,
                "asd_interp_with_coeff: get workspace size failed.");

    // 准备执行流和 Workspace 内存
    auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor workspace_tensor;
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        workspace_tensor = at_npu::native::allocate_workspace(static_cast<long>(workspace_size), sip_stream);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    EXEC_FUNC(AsdSip::asdInterpWithCoeff, acl_x, acl_coefficient, acl_output, sip_stream, workspace_addr);

    return output;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_interp_with_coeff(Tensor x, Tensor coefficient) -> Tensor"); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_interp_with_coeff", &asdInterpWithCoeff); }

} // namespace sip_pta
