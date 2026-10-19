/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/assert.h"
#include "log/log.h"
#include "base_inner_api.h"
#include "utils/ops_base.h"
#include "aclnnop/aclnn_flip.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {
AspbStatus Reverse(const Tensor& inTensor, Tensor& outTensor, SVector<int64_t> axis, void* stream,
                   uint8_t* deviceBuffer)
{
    outTensor.desc = inTensor.desc;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* inAclTensor = nullptr;
    aclTensor* outAclTensor = nullptr;
    toAclTensor(inTensor, inAclTensor);
    if (inAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "Fail to convert in mki tensor to aclTensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(outTensor, outAclTensor);
    if (outAclTensor == nullptr) {
        aclDestroyTensor(inAclTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert out mki tensor to aclTensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    auto dims = aclCreateIntArray(axis.data(), axis.size());

    auto ret = aclnnFlipGetWorkspaceSize(inAclTensor, dims, outAclTensor, &workspaceSize, &executor);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnFlipGetWorkspaceSize failed. ERROR: " << ret;
        aclDestroyTensor(inAclTensor);
        aclDestroyTensor(outAclTensor);
        aclDestroyIntArray(dims);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
            executor = nullptr;
        }
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    ret = aclnnFlip(deviceBuffer, workspaceSize, executor, stream);
    if (ret != ::ACL_SUCCESS) {
        aclDestroyAclOpExecutor(executor);
        ASDSIP_LOG(ERROR) << "aclnnFlip failed. ERROR: " << ret;
        aclDestroyTensor(inAclTensor);
        aclDestroyTensor(outAclTensor);
        aclDestroyIntArray(dims);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    aclDestroyAclOpExecutor(executor);
    aclDestroyTensor(inAclTensor);
    aclDestroyTensor(outAclTensor);
    aclDestroyIntArray(dims);
    ASDSIP_LOG(INFO) << "Execute Reverse success.";
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
