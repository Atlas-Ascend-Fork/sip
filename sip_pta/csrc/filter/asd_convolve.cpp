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

#include "filter_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace sip_pta {

/**
 * @brief 一维滤波卷积算子 (Convolve)
 */
at::Tensor asdConvolve(const at::Tensor& signal, const at::Tensor& kernel)
{
    c10_npu::NPUGuard guard(signal.device());

    // 基础参数校验：信号为复数，卷积核为实数
    TORCH_CHECK(signal.scalar_type() == at::kComplexFloat || signal.scalar_type() == at::kComplexHalf,
                "asd_convolve: Signal must be ComplexFloat or ComplexHalf.");
    TORCH_CHECK(kernel.scalar_type() == at::kFloat || kernel.scalar_type() == at::kHalf,
                "asd_convolve: Kernel must be Float32 or Float16.");

    // 维度校验：信号为 [Batch, n]，卷积核为 [k]
    TORCH_CHECK(signal.dim() == sip_pta::DIM_2, "asd_convolve: Signal must be a 2D tensor [BatchCount, n].");
    TORCH_CHECK(kernel.dim() == sip_pta::DIM_1, "asd_convolve: Kernel must be a 1D tensor [k].");

    int64_t signalLen = signal.size(1);
    int64_t kernelLen = kernel.size(0);
    int64_t batchCount = signal.size(0);

    // 信号长度限制
    static constexpr int64_t MIN_SIGNAL_LEN = 12;
    static constexpr int64_t MAX_SIGNAL_LEN = 26208;

    // 卷积核长度限制
    static constexpr int64_t MIN_KERNEL_LEN = 8;
    static constexpr int64_t MAX_KERNEL_LEN = 32;

    // Batch 限制
    static constexpr int64_t MIN_BATCH = 1;
    static constexpr int64_t MAX_BATCH = 768;

    TORCH_CHECK(signalLen >= MIN_SIGNAL_LEN && signalLen <= MAX_SIGNAL_LEN, "asd_convolve: signal length must be in [",
                MIN_SIGNAL_LEN, ", ", MAX_SIGNAL_LEN, "], but got ", signalLen);

    TORCH_CHECK(kernelLen >= MIN_KERNEL_LEN && kernelLen <= MAX_KERNEL_LEN, "asd_convolve: kernel length must be in [",
                MIN_KERNEL_LEN, ", ", MAX_KERNEL_LEN, "], but got ", kernelLen);

    TORCH_CHECK(batchCount >= MIN_BATCH && batchCount <= MAX_BATCH, "asd_convolve: batch count must be in [", MIN_BATCH,
                ", ", MAX_BATCH, "], but got ", batchCount);
    // 输出 shape 与输入 signal 保持一致
    at::Tensor output = at::empty_like(signal);

    // 获取 Workspace 大小（检查返回值，口径对齐 #259 修复后的 rs_interpolation，issue #286）
    size_t workspace_size = 0;
    TORCH_CHECK(
        AsdSip::asdConvolveGetWorkspaceSize(signalLen, kernelLen, workspace_size) == AsdSip::ErrorType::ACL_SUCCESS,
        "asd_convolve: get workspace size failed.");

    auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor workspace_tensor;
    void* workspace_addr = nullptr;

    if (workspace_size > 0) {
        workspace_tensor = at_npu::native::allocate_workspace(static_cast<long>(workspace_size), sip_stream);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    asdConvolveMode_t mode = AsdSip::asdConvolveMode_t::ASD_CONVOLVE_SAME;
    aclTensor* acl_signal = CreateAclTensorFromAtTensor(signal);
    aclTensor* acl_kernel = CreateAclTensorFromAtTensor(kernel);
    aclTensor* acl_output = CreateAclTensorFromAtTensor(output);

    // 调用算子，默认传入 SAME 模式
    EXEC_FUNC(AsdSip::asdConvolve, acl_signal, acl_kernel, acl_output, mode, sip_stream, workspace_addr);

    return output;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_convolve(Tensor signal, Tensor kernel) -> Tensor"); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_convolve", &asdConvolve); }

} // namespace sip_pta
