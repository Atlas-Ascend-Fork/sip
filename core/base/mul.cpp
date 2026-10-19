/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "base_inner_api.h"
#include "utils/assert.h"
#include "utils/ops_base.h"
#include "log/log.h"
#include "aclnnop/aclnn_mul.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {

AsdSip::AspbStatus Mul(const Mki::Tensor& inputA, const Mki::Tensor& inTensorB, Mki::Tensor& outTensor, void* stream,
                       uint8_t* deviceBuffer)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* mulAAclTensor = nullptr;
    aclTensor* mulBAclTensor = nullptr;
    aclTensor* outAclTensor = nullptr;

    toAclTensor(inputA, mulAAclTensor);
    if (mulAAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "Fail to convert in mki A tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(inTensorB, mulBAclTensor);
    if (mulBAclTensor == nullptr) {
        aclDestroyTensor(mulAAclTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert in mki B tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(outTensor, outAclTensor);
    if (outAclTensor == nullptr) {
        aclDestroyTensor(mulAAclTensor);
        aclDestroyTensor(mulBAclTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert out mki out tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    auto ret = aclnnMulGetWorkspaceSize(mulAAclTensor, mulBAclTensor, outAclTensor, &workspaceSize, &executor);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnMulGetWorkspaceSize failed. ERROR: " << ret;
        aclDestroyTensor(mulAAclTensor);
        aclDestroyTensor(mulBAclTensor);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    ret = aclnnMul(deviceBuffer, workspaceSize, executor, stream);
    if (ret != ::ACL_SUCCESS) {
        aclDestroyAclOpExecutor(executor);
        ASDSIP_LOG(ERROR) << "aclnnMul failed. ERROR: " << ret;
        aclDestroyTensor(mulAAclTensor);
        aclDestroyTensor(mulBAclTensor);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    aclDestroyAclOpExecutor(executor);
    aclDestroyTensor(mulAAclTensor);
    aclDestroyTensor(mulBAclTensor);
    aclDestroyTensor(outAclTensor);
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
