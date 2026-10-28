/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/math/math.h>
#include <mki_loader/op_register.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"

#include "ifftshift.h"
#include "tiling/ifftshift_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_CNT = 1;
static constexpr uint32_t TENSOR_OUTPUT_CNT = 1;
namespace AsdSip {

using Mki::AnyCast;
using Mki::BinHandle;
using Mki::KernelBase;
using Mki::KernelInfo;
using Mki::LaunchParam;
using Mki::Status;
using Mki::TENSOR_DTYPE_COMPLEX64;

class IfftShiftKernel : public KernelBase {
public:
    explicit IfftShiftKernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : KernelBase(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_CNT, "check inTensor count failed", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_CNT, "check outTensor count failed",
                     return false);

        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::IfftShift), "check param type failed!",
                     return false);
        const auto& param = AnyCast<OpParam::IfftShift>(launchParam.GetParam());
        ASDSIP_CHECK(param.row > 0 && param.col > 0 && param.batch > 0, "check shape failed!", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return sizeof(IfftShiftTilingData);
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = IfftShiftTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitKernelInfoImpl IfftShiftTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

class IfftShiftC64Kernel : public IfftShiftKernel {
public:
    explicit IfftShiftC64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : IfftShiftKernel(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(IfftShiftKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "inTensor 0 dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "outTensor 0 dtype unsupported",
                     return false);
        return true;
    }
};

REG_KERNEL_BASE(IfftShiftC64Kernel);

} // namespace AsdSip
