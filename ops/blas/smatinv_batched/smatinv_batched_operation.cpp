/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/base/operation_base.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mki_loader/op_register.h"
#include "mki/utils/SVector/SVector.h"
#include "smatinv_batched.h"

static constexpr int64_t INPUT_NUM = 2;
static constexpr int64_t OUTPUT_NUM = 1;

namespace AsdSip {
using namespace Mki;
class SmatinvBatchedOperation : public OperationBase {
public:
    explicit SmatinvBatchedOperation(const std::string& opName) noexcept : OperationBase(opName) {}
    Kernel* GetBestKernel(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::SmatinvBatched), "OpParam is invalid",
                     return nullptr);

        // 单 bin 内部按 tiling.dtype 分支 fp32/fp16（对齐 CmatinvBatchedC64Kernel 模式）
        return GetKernelByName("SmatinvBatchedF32Kernel");
    }

    int64_t GetInputNum(const Any& specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::SmatinvBatched), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

    int64_t GetOutputNum(const Any& specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::SmatinvBatched), "OpParam is invalid", return 0);
        return OUTPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launchParam, SVector<Tensor>& outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::SmatinvBatched), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INVALID_VALUE));

        outTensors[0].desc = launchParam.GetOutTensor(0).desc;

        return Status::OkStatus();
    }
};
REG_OPERATION(SmatinvBatchedOperation);
} // namespace AsdSip
