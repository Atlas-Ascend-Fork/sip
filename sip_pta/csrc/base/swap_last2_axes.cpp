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

#include "base_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>
#include <tuple>

namespace sip_pta {
at::Tensor swapLast2Axes(const at::Tensor& input)
{
    c10_npu::NPUGuard guard(input.device());

    int64_t dim = input.dim();
    TORCH_CHECK(dim >= sip_pta::DIM_2, "swap_last_two_axes expects input with at least 2 dimensions, but got ", dim);
    auto sizes = input.sizes().vec();
    std::swap(sizes[dim - sip_pta::DIM_1], sizes[dim - sip_pta::DIM_2]);
    at::Tensor output = at::empty(sizes, input.options());
    size_t workspace_size = 0;
    TORCH_CHECK(AsdSip::swapLast2AxesGetWorkspaceSize(workspace_size) == AsdSip::ErrorType::ACL_SUCCESS,
                "swap_last2_axes: get workspace size failed.");

    auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor workspace_tensor;
    void* workspace_addr = nullptr;
    if (workspace_size != 0) {
        workspace_tensor = at_npu::native::allocate_workspace(static_cast<long>(workspace_size), sip_stream);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }
    aclTensor* acl_in = CreateAclTensorFromAtTensor(input);
    aclTensor* acl_out = CreateAclTensorFromAtTensor(output);
    EXEC_FUNC(AsdSip::swapLast2Axes, acl_in, acl_out, sip_stream, workspace_addr);
    return output;
}
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("swap_last2_axes(Tensor x) -> Tensor"); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("swap_last2_axes", &swapLast2Axes); }
} // namespace sip_pta
