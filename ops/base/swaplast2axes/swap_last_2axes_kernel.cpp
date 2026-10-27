/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/base/kernel_base.h"
#include "swap_last_2axes.h"
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/math/math.h"
#include "mki_loader/op_register.h"
#include "utils/assert.h"
#include "log/log.h"
#include "tiling/swap_last_2axes_tiling_data.h"
#include "tiling/swap_last_2axes_tiling.h"

namespace AsdSip {
using namespace Mki;
constexpr uint64_t TENSOR_PERM_IDX = 1;

class SwapLast2AxesKernel : public KernelBase {
public:
    explicit SwapLast2AxesKernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : KernelBase(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::SwapLast2Axes),
                     "SwapLast2Axes: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return sizeof(SwapLast2AxesTilingData);
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = SwapLast2AxesTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl SwapLast2AxesTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// SwapLast2AxesC64Kernel
class SwapLast2AxesC64Kernel : public SwapLast2AxesKernel {
public:
    explicit SwapLast2AxesC64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : SwapLast2AxesKernel(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(SwapLast2AxesKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                     return false);
        return true;
    }
};
REG_KERNEL_BASE(SwapLast2AxesC64Kernel);

} // namespace AsdSip
