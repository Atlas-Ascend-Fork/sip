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

#include "domain/rs_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace sip_pta {

/**
 * @brief 批量一维复数向量 Sinc 插值算子
 */
void checkSincInterpolationParams(const at::Tensor& inputTensor, const at::Tensor& sincTab, const at::Tensor& posFloor,
                                  const at::Tensor& posToTabIndex, int64_t interpNum, int64_t quantNum)
{
    // 1. 数据类型校验
    TORCH_CHECK(inputTensor.scalar_type() == at::kComplexFloat, "inputTensor must be ComplexFloat.");
    TORCH_CHECK(sincTab.scalar_type() == at::kFloat, "sincTab must be Float32.");
    TORCH_CHECK(posFloor.scalar_type() == at::kInt, "posFloor must be Int32.");
    TORCH_CHECK(posToTabIndex.scalar_type() == at::kShort, "posToTabIndex must be Int16.");

    // 2. 维度校验
    TORCH_CHECK(inputTensor.dim() == sip_pta::DIM_2, "inputTensor must be 2D.");
    TORCH_CHECK(posFloor.dim() == sip_pta::DIM_2 && posToTabIndex.dim() == sip_pta::DIM_2,
                "posFloor and posToTabIndex must be 2D.");

    // 3. 形状匹配校验
    const int64_t batch = inputTensor.size(0);
    const int64_t posLength = posFloor.size(1);
    TORCH_CHECK(posFloor.size(0) == batch && posToTabIndex.size(0) == batch, "Batch sizes of inputs must match.");
    TORCH_CHECK(posToTabIndex.size(1) == posLength, "posLength of posFloor and posToTabIndex must match.");

    // 4. 算子规格约束
    static constexpr int64_t MAX_INTERP_NUM = 16;
    static constexpr int64_t MAX_QUANT_NUM = 32;
    static constexpr int64_t EVEN_DIVISOR = 2;

    // 下界校验补充（issue #258）：原仅校验上界，0 与负偶数（-2 % 2 == 0）及
    // quantNum=0（0 & -1 == 0 恒成立）均可穿透
    TORCH_CHECK(interpNum >= 2 && interpNum <= MAX_INTERP_NUM && interpNum % EVEN_DIVISOR == 0,
                "interpNum must be an even number in [2, 16], got ", interpNum);
    TORCH_CHECK(quantNum >= 1 && quantNum <= MAX_QUANT_NUM && (quantNum & (quantNum - 1)) == 0,
                "quantNum must be a power of 2 in [1, 32], got ", quantNum);

    // 5. sincTab 容量校验
    const int64_t expectedTabSize = 4 * ((quantNum + 1) * 2) * (interpNum * 2 + 8);
    TORCH_CHECK(sincTab.numel() == expectedTabSize, "sincTab numel must be exactly ", expectedTabSize);
}

at::Tensor rsInterpolationBySinc(const at::Tensor& inputTensor, const at::Tensor& sincTab, const at::Tensor& posFloor,
                                 const at::Tensor& posToTabIndex, int64_t interpNum, int64_t quantNum)
{
    c10_npu::NPUGuard guard(inputTensor.device());

    checkSincInterpolationParams(inputTensor, sincTab, posFloor, posToTabIndex, interpNum, quantNum);

    const int64_t batch = inputTensor.size(0);
    const int64_t posLength = posFloor.size(1);
    at::Tensor outputTensor = at::empty({batch, posLength}, inputTensor.options());

    size_t workspace_size = 0;
    // 返回值检查（issue #259，对齐同目录其他适配口径）：失败时以 0 继续会传入
    // nullptr workspace 且吞掉原始错误
    TORCH_CHECK(AsdSip::rsInterpolationBySincGetWorkspaceSize(workspace_size) == AsdSip::ErrorType::ACL_SUCCESS,
                "rsInterpolationBySincGetWorkspaceSize failed.");
    auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);

    void* workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size > 0) {
        workspace_tensor = at_npu::native::allocate_workspace(static_cast<long>(workspace_size), sip_stream);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    const int interpNum_i = static_cast<int>(interpNum);
    const int quantNum_i = static_cast<int>(quantNum);
    const int interpLength_i = static_cast<int>(posLength);
    auto aclInput = CreateAclTensorFromAtTensor(inputTensor);
    auto aclSincTab = CreateAclTensorFromAtTensor(sincTab);
    auto aclPosFloor = CreateAclTensorFromAtTensor(posFloor);
    auto aclPosToTabIndex = CreateAclTensorFromAtTensor(posToTabIndex);
    auto aclOutput = CreateAclTensorFromAtTensor(outputTensor);

    EXEC_FUNC(AsdSip::rsInterpolationBySinc, aclInput, aclSincTab, aclPosFloor, aclPosToTabIndex, aclOutput,
              interpNum_i, quantNum_i, interpLength_i, sip_stream, workspace_addr);

    return outputTensor;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("rs_interpolation_by_sinc(Tensor inputTensor, Tensor sincTab, Tensor posFloor, Tensor "
          "posToTabIndex, int "
          "interpNum, int quantNum) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("rs_interpolation_by_sinc", &rsInterpolationBySinc); }
} // namespace sip_pta
