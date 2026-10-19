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
#include "aclnnop/aclnn_div.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {

AsdSip::AspbStatus Div(const Mki::Tensor& inTensorA, const Mki::Tensor& inTensorB, Mki::Tensor& outTensor, void* stream,
                       uint8_t* deviceBuffer)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* divBAclTensor = nullptr;
    aclTensor* divBclTensor = nullptr;
    aclTensor* outAclTensor = nullptr;

    toAclTensor(inTensorA, divBAclTensor);
    if (divBAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "Fail to convert in mki A tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(inTensorB, divBclTensor);
    if (divBclTensor == nullptr) {
        aclDestroyTensor(divBAclTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert in mki B tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(outTensor, outAclTensor);
    if (outAclTensor == nullptr) {
        aclDestroyTensor(divBAclTensor);
        aclDestroyTensor(divBclTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert out mki out tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    auto ret = aclnnDivGetWorkspaceSize(divBAclTensor, divBclTensor, outAclTensor, &workspaceSize, &executor);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnDivGetWorkspaceSize failed. ERROR: " << ret;
        aclDestroyTensor(divBAclTensor);
        aclDestroyTensor(divBclTensor);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    ret = aclnnDiv(deviceBuffer, workspaceSize, executor, stream);
    if (ret != ::ACL_SUCCESS) {
        aclDestroyAclOpExecutor(executor);
        ASDSIP_LOG(ERROR) << "aclnnDiv failed. ERROR: " << ret;
        aclDestroyTensor(divBAclTensor);
        aclDestroyTensor(divBclTensor);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    aclDestroyAclOpExecutor(executor);
    aclDestroyTensor(divBAclTensor);
    aclDestroyTensor(divBclTensor);
    aclDestroyTensor(outAclTensor);
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
