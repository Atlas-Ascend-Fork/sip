/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include "mki/base/operation_base.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mki_loader/op_register.h"
#include "mki/utils/SVector/SVector.h"
#include "sgetri_batched.h"

static constexpr int64_t INPUT_NUM = 8;
static constexpr int64_t OUTPUT_NUM = 1;

namespace AsdSip {
using namespace Mki;
class SgetriBatchedOperation : public OperationBase {
public:
    explicit SgetriBatchedOperation(const std::string& opName) noexcept : OperationBase(opName) {}
    Kernel* GetBestKernel(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::SgetriBatched), "OpParam is invalid",
                     return nullptr);
        /*
         * P2 分流 (每核独立认领矩阵全流程, 零跨核同步, 收益随 batch 放大):
         * n<=256 且 tail 合法且 (batchSize>=2 || n<=128) 走纯 AIV kernel
         * SgetriP2BatchedF32Kernel; 其余走 MIX 主路径 SgetriBatchedF32Kernel。
         * tail 合法性约束: 末面板有效列宽 realN*4B 须 32B 对齐 (VEC 指令
         * 最小块), 即 n%16∈{0,8}; 其余 tail 走 MIX。
         */
        OpParam::SgetriBatched param = AnyCast<OpParam::SgetriBatched>(launchParam.GetParam());
        // 调试覆盖开关: ASDSIP_SGETRI_P2=1 强制全 shape 走 P2; =0 强制全走
        // MIX; 未设置时按上述规则分流。
        static const char* p2Env = getenv("ASDSIP_SGETRI_P2");
        static const bool p2Force = (p2Env != nullptr && p2Env[0] == '1');
        static const bool mixForce = (p2Env != nullptr && p2Env[0] == '0');
        const bool tailOk = (param.n % 16 == 0) || (param.n % 16 == 8);
        if (p2Force || (!mixForce && param.n <= 256 && tailOk && (param.batchSize >= 2 || param.n <= 128))) {
            return GetKernelByName("SgetriP2BatchedF32Kernel");
        }
        return GetKernelByName("SgetriBatchedF32Kernel");
    }

    int64_t GetInputNum(const Any& specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::SgetriBatched), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

    int64_t GetOutputNum(const Any& specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::SgetriBatched), "OpParam is invalid", return 0);
        return OUTPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launchParam, SVector<Tensor>& outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::SgetriBatched), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INVALID_VALUE));

        outTensors[0].desc = launchParam.GetOutTensor(0).desc;

        return Status::OkStatus();
    }
};
REG_OPERATION(SgetriBatchedOperation);
} // namespace AsdSip
