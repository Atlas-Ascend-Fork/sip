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

namespace sip_pta {
at::Tensor asdMul(const at::Tensor& x, const at::Tensor& y)
{
    // 一致性校验（issue #384，对齐 solve/#372 口径）：x/y 形状、dtype、设备必须一致，
    // 否则内核按 x.numel() 解读 y 内存——y 元素数不足时设备侧越界读，dtype/设备不一致
    // 时按错误元素宽度或地址空间解引用，结果静默错误
    TORCH_CHECK(x.sizes() == y.sizes(), "asd_mul: x and y must have the same shape.");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "asd_mul: x and y must have the same dtype.");
    TORCH_CHECK(x.device() == y.device(), "asd_mul: x and y must be on the same device.");

    c10_npu::NPUGuard guard(x.device());

    at::Tensor z = at::empty_like(x);

    auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t n = x.numel();
    uint8_t* workspace_ptr = nullptr;
    EXEC_FUNC(AsdSip::asdMul, n, x, y, z, sip_stream, workspace_ptr);
    return z;
}
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_mul(Tensor x, Tensor y) -> Tensor"); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_mul", &asdMul); }
} // namespace sip_pta
