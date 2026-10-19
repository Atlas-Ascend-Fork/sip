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
#include "aclnnop/aclnn_mul.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {

AsdSip::AspbStatus Muls(const Mki::Tensor& inTensorA, float value, Mki::Tensor& outTensor, void* stream,
                        uint8_t* deviceBuffer)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* inAclTensor = nullptr;
    aclTensor* outAclTensor = nullptr;
    aclScalar* other = nullptr;

    toAclTensor(inTensorA, inAclTensor);
    if (inAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "Fail to convert in mki A tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(outTensor, outAclTensor);
    if (outAclTensor == nullptr) {
        aclDestroyTensor(inAclTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert out mki out tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    other = aclCreateScalar(&value, aclDataType::ACL_FLOAT);
    if (other == nullptr) {
        aclDestroyTensor(inAclTensor);
        aclDestroyTensor(outAclTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert out scalar to aclScalar.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    auto ret = aclnnMulsGetWorkspaceSize(inAclTensor, other, outAclTensor, &workspaceSize, &executor);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnMulsGetWorkspaceSize failed. ERROR: " << ret;
        aclDestroyTensor(inAclTensor);
        aclDestroyScalar(other);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    ret = aclnnMuls(deviceBuffer, workspaceSize, executor, stream);
    if (ret != ::ACL_SUCCESS) {
        aclDestroyAclOpExecutor(executor);
        ASDSIP_LOG(ERROR) << "aclnnMuls failed. ERROR: " << ret;
        aclDestroyTensor(inAclTensor);
        aclDestroyScalar(other);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    aclDestroyAclOpExecutor(executor);
    aclDestroyTensor(inAclTensor);
    aclDestroyScalar(other);
    aclDestroyTensor(outAclTensor);
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
