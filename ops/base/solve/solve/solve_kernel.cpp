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
#include "mki_loader/op_register.h"
#include "utils/assert.h"
#include "log/log.h"
#include "solve.h"
#include "tiling/solve_tiling.h"
#include "tiling/solve_tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AsdSip {
using namespace Mki;
class SolveKernel : public KernelBase {
public:
    explicit SolveKernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : KernelBase(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "check outTensor count failed",
                     return false);
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Solve), "check param type failed!", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return sizeof(SolveTilingData);
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = SolveTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl SolveTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

class SolveFp32Kernel : public SolveKernel {
public:
    explicit SolveFp32Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : SolveKernel(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(SolveKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "A tensor dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT, "B tensor dtype unsupported",
                     return false);
        return true;
    }
};
REG_KERNEL_BASE(SolveFp32Kernel);

} // namespace AsdSip
