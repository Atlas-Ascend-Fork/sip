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
#include "aclnnop/aclnn_copy.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {

AspbStatus AsStrided(const Tensor& inTensor, Tensor& outTensor, SVector<int64_t> size, SVector<int64_t> stride,
                     void* stream, int64_t offset, uint8_t* deviceBuffer)
{
    Tensor tmpTensor = inTensor;

    tmpTensor.desc.dims = size;
    tmpTensor.desc.offset = offset;

    auto strideData = stride.data();
    std::vector<int64_t> strideVec(stride.size());
    strideVec.assign(strideData, strideData + stride.size());

    outTensor.desc = {tmpTensor.desc.dtype, tmpTensor.desc.format, size, {}, 0};

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* inAclTensor = nullptr;
    aclTensor* outAclTensor = nullptr;
    toAclTensor(tmpTensor, inAclTensor, strideVec);
    if (inAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "Fail to convert in mki tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(outTensor, outAclTensor);
    if (outAclTensor == nullptr) {
        aclDestroyTensor(inAclTensor);
        ASDSIP_LOG(ERROR) << "Fail to convert out mki tensor to acltensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    auto ret = aclnnInplaceCopyGetWorkspaceSize(outAclTensor, inAclTensor, &workspaceSize, &executor);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnInplaceCopyGetWorkspaceSize failed. ERROR: " << ret;
        aclDestroyTensor(inAclTensor);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    ret = aclnnInplaceCopy(deviceBuffer, workspaceSize, executor, stream);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnInplaceCopy failed. ERROR: " << ret;
        aclDestroyTensor(inAclTensor);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    // executor 不由本函数释放（回归记录，PR #200 复测）：aclnnInplaceCopy 为
    // inplace 变体，二段接口执行时已接管并释放 executor 内部资源，调用方再
    // aclDestroyAclOpExecutor 会触发 ~aclTensorList double free（UT 2D C2C
    // 矩形用例 Transpose::Run → AsStrided 路径 SIGABRT 复现）。与普通 aclnn
    // 算子（mul/slice 等调用方 destroy）的所有权语义不同，故此处保持不 destroy。
    aclDestroyTensor(inAclTensor);
    aclDestroyTensor(outAclTensor);
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
