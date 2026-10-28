/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ifftshift.h"

namespace AsdSip {

using Mki::Any;
using Mki::AnyCast;
using Mki::Kernel;
using Mki::LaunchParam;
using Mki::OperationBase;
using Mki::Status;
using Mki::SVector;
using Mki::Tensor;

constexpr int64_t INPUT_NUM = 1;

class IfftShiftOperation : public OperationBase {
public:
    explicit IfftShiftOperation(const std::string& opName) noexcept : OperationBase(opName) {}
    Kernel* GetBestKernel(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);

        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            // Single unified kernel: the runtime-col 3-stage fast pipeline
            // covers every column whose rows fit the per-slot UB budget
            // (col <= ~7800, incl. the fixed MRI col == 50), wider columns
            // fall back to the piecewise path inside the same kernel.
            return GetKernelByName("IfftShiftC64Kernel");
        }

        ASDSIP_LOG(ERROR) << "Unsupported platform type for IfftShiftOperation";
        return nullptr;
    }

    int64_t GetInputNum(const Any& specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::IfftShift), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launchParam, SVector<Tensor>& outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::IfftShift), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INVALID_VALUE));

        const auto& param = AnyCast<OpParam::IfftShift>(launchParam.GetParam());
        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dims = {param.batch, param.row, param.col};

        return Status::OkStatus();
    }
};
REG_OPERATION(IfftShiftOperation);
} // namespace AsdSip
