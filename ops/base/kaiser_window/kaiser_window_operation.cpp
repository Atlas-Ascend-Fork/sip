/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/operation_base.h>
#include <mki/utils/platform/platform_info.h>
#include "mki_loader/op_register.h"
#include "params/kaiser_window.h"
#include "utils/assert.h"

namespace AsdSip {
using namespace Mki;
class KaiserWindowOperation : public OperationBase {
public:
    explicit KaiserWindowOperation(const std::string& name) noexcept : OperationBase(name) {}
    int64_t GetInputNum(const Any&) const override { return 0; }
    int64_t GetOutputNum(const Any&) const override { return 1; }

    Kernel* GetBestKernel(const LaunchParam& launch) const override
    {
        ASDSIP_CHECK(IsConsistent(launch), "Inconsistent KaiserWindow launch", return nullptr);
        const auto platform = PlatformInfo::Instance().GetPlatformType();
        if (platform == PlatformType::ASCEND_950) {
            return GetKernelByName("KaiserWindowArch35Kernel");
        }
        if (platform == PlatformType::ASCEND_910B) {
            return GetKernelByName("KaiserWindowKernel");
        }
        return nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launch, SVector<Tensor>& outputs) const override
    {
        ASDSIP_CHECK(launch.GetParam().Type() == typeid(OpParam::KaiserWindow) && outputs.size() == 1,
                     "Invalid KaiserWindow parameter", return Status::FailStatus(ERROR_INVALID_VALUE));
        const auto& param = AnyCast<OpParam::KaiserWindow>(launch.GetParam());
        outputs[0].desc.dtype = static_cast<TensorDType>(param.dtype);
        outputs[0].desc.format = TENSOR_FORMAT_ND;
        outputs[0].desc.dims = {param.windowLength};
        return Status::OkStatus();
    }
};
REG_OPERATION(KaiserWindowOperation);
} // namespace AsdSip
