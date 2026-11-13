/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <limits>
#include <mki/base/kernel_base.h>
#include "mki_loader/op_register.h"
#include "params/kaiser_window.h"
#include "tiling/kaiser_window_tiling.h"
#include "tiling/tiling_data.h"
#include "utils/assert.h"

namespace AsdSip {
using namespace Mki;
class KaiserWindowKernel : public KernelBase {
public:
    KaiserWindowKernel(const std::string& name, const BinHandle* handle) noexcept : KernelBase(name, handle) {}
    bool CanSupport(const LaunchParam& launch) const override
    {
        if (launch.GetInTensorCount() != 0 || launch.GetOutTensorCount() != 1 ||
            launch.GetParam().Type() != typeid(OpParam::KaiserWindow)) {
            return false;
        }
        const auto& p = AnyCast<OpParam::KaiserWindow>(launch.GetParam());
        const auto& out = launch.GetOutTensor(0);
        return p.windowLength >= 0 && p.windowLength <= std::numeric_limits<int32_t>::max() &&
               out.desc.dims.size() == 1 && out.desc.dims[0] == p.windowLength && out.desc.dtype == p.dtype &&
               out.IsContiguous() &&
               (p.dtype == TENSOR_DTYPE_FLOAT || p.dtype == TENSOR_DTYPE_FLOAT16 || p.dtype == TENSOR_DTYPE_BF16);
    }
    uint64_t GetTilingSize(const LaunchParam&) const override { return sizeof(KaiserWindowTilingData); }
    Status InitImpl(const LaunchParam& launch) override
    {
        ASDSIP_CHECK(CanSupport(launch), "Invalid KaiserWindow launch", return Status::FailStatus(ERROR_INVALID_VALUE));
        ASDSIP_CHECK(KaiserWindowTiling(launch, kernelInfo_) == ErrorType::ACL_SUCCESS, "KaiserWindow tiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};
REG_KERNEL_BASE(KaiserWindowKernel);
class KaiserWindowArch35Kernel : public KaiserWindowKernel {
public:
    KaiserWindowArch35Kernel(const std::string& name, const BinHandle* handle) noexcept
        : KaiserWindowKernel(name, handle)
    {}
};
REG_KERNEL_BASE(KaiserWindowArch35Kernel);
} // namespace AsdSip
