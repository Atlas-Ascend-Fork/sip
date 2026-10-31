
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
#include "aclnnop/aclnn_cat.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {
AspbStatus Concat(const Tensor& inTensorA, const Tensor& inTensorB, Tensor& outTensor, int concatDim, void* stream,
                  uint8_t* deviceBuffer)
{
    SVector<int64_t> dims = inTensorA.desc.dims;
    const SVector<int64_t>& dimsB = inTensorB.desc.dims;

    ASDSIP_ECHECK(dims.size() == dimsB.size(), "Input tensors A and B must have the same number of dimensions",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int32_t dimSize = static_cast<int32_t>(dims.size());
    for (int32_t i = 0; i < dimSize; i++) {
        if (i != concatDim && dims.at(i) != dimsB.at(i)) {
            ASDSIP_LOG(ERROR) << "Non-concatenated dimensions are inconsistent.";
            return ACL_ERROR_INTERNAL_ERROR;
        }
    }
    dims.at(concatDim) = dims.at(concatDim) + dimsB.at(concatDim);
    outTensor.desc = {inTensorA.desc.dtype, inTensorA.desc.format, dims, {}, 0};

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* inAclTensorA = nullptr;
    aclTensor* inAclTensorB = nullptr;
    aclTensor* outAclTensor = nullptr;
    toAclTensor(inTensorA, inAclTensorA);
    if (inAclTensorA == nullptr) {
        ASDSIP_LOG(ERROR) << "Fail to convert in mki A tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(inTensorB, inAclTensorB);
    if (inAclTensorB == nullptr) {
        aclDestroyTensor(inAclTensorA);
        ASDSIP_LOG(ERROR) << "Fail to convert in mki B tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    std::vector<aclTensor*> tmp{inAclTensorA, inAclTensorB};
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    toAclTensor(outTensor, outAclTensor);
    if (outAclTensor == nullptr) {
        aclDestroyTensorList(tensorList);
        ASDSIP_LOG(ERROR) << "Fail to convert out mki out tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    auto ret = aclnnCatGetWorkspaceSize(tensorList, concatDim, outAclTensor, &workspaceSize, &executor);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnCatGetWorkspaceSize failed. ERROR: " << ret;
        aclDestroyTensorList(tensorList);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    ret = aclnnCat(deviceBuffer, workspaceSize, executor, stream);
    if (ret != ::ACL_SUCCESS) {
        aclDestroyAclOpExecutor(executor);
        ASDSIP_LOG(ERROR) << "aclnnCat failed. ERROR: " << ret;
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
