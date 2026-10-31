
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
#include "aclnnop/aclnn_stack.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {
AspbStatus MakeComplex(const Tensor& realTensor, const Tensor& imagTensor, Tensor& outTensor, int64_t dim, void* stream,
                       uint8_t* deviceBuffer)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* aclRealTensor = nullptr;
    aclTensor* aclImagTensor = nullptr;
    aclTensor* outAclTensor = nullptr;
    toAclTensor(realTensor, aclRealTensor);
    if (aclRealTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "Fail to convert real mki tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(imagTensor, aclImagTensor);
    if (aclImagTensor == nullptr) {
        aclDestroyTensor(aclRealTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert imag mki tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    std::vector<aclTensor*> tmp{aclRealTensor, aclImagTensor};
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    toAclTensor(outTensor, outAclTensor);
    if (outAclTensor == nullptr) {
        aclDestroyTensorList(tensorList);
        ASDSIP_LOG(ERROR) << "Fail to convert out mki out tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    auto ret = aclnnStackGetWorkspaceSize(tensorList, dim, outAclTensor, &workspaceSize, &executor);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnStackGetWorkspaceSize failed. ERROR: " << ret;
        aclDestroyTensorList(tensorList);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    ret = aclnnStack(deviceBuffer, workspaceSize, executor, stream);
    if (ret != ::ACL_SUCCESS) {
        aclDestroyAclOpExecutor(executor);
        ASDSIP_LOG(ERROR) << "aclnnStack failed. ERROR: " << ret;
        aclDestroyTensorList(tensorList);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    aclDestroyAclOpExecutor(executor);
    aclDestroyTensorList(tensorList);
    aclDestroyTensor(outAclTensor);
    ASDSIP_LOG(INFO) << "Execute Concat success.";
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
