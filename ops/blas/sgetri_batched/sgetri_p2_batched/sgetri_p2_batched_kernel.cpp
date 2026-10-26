/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/math/math.h"
#include "mki_loader/op_register.h"
#include "sgetri_batched.h"
#include "utils/assert.h"
#include "log/log.h"
#include "tiling/sgetri_p2_batched_tiling_data.h"
#include "tiling/sgetri_p2_batched_tiling.h"

static constexpr int64_t INPUT_NUM = 8;
static constexpr int64_t OUTPUT_NUM = 1;

namespace AsdSip {

/*
 * P2 纯 AIV kernel (每核独立矩阵全流程, 见 kernel/sgetri_p2_batched.cpp)。
 * 与 MIX 版共用 OpParam::SgetriBatched, tiling 布局同名结构
 * (SgetriP2BatchedTilingData), blockDim = vector 核数 40。
 */
class SgetriP2BatchedF32Kernel : public KernelBase {
public:
    explicit SgetriP2BatchedF32Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : KernelBase(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::SgetriBatched),
                     "SgetriP2Batched: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == INPUT_NUM, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == OUTPUT_NUM, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return sizeof(SgetriP2BatchedTilingData);
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = SgetriP2BatchedTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitImpl SgetriP2BatchedTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

REG_KERNEL_BASE(SgetriP2BatchedF32Kernel);
} // namespace AsdSip
