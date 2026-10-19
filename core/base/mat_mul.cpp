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
#include "aclnnop/aclnn_matmul.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {
AspbStatus MatMul(const Tensor& inATensor, const Tensor& inBTensor, Tensor& outTensor, int m, int k, int n,
                  void* stream, uint8_t* deviceBuffer)
{
    ASDSIP_CHECK(m > 0 && k > 0, "check inTensor dims failed", return ACL_ERROR_INVALID_PARAM);

    SVector<int64_t> dims = inATensor.desc.dims;
    ASDSIP_CHECK(dims.size() > 0, "check inTensor dims failed", return ACL_ERROR_INVALID_PARAM);

    dims.at(dims.size() - 1) = n;
    outTensor.desc = {inATensor.desc.dtype, inATensor.desc.format, dims, {}, 0};
    int8_t cubeMathType = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* inAAclTensor = nullptr;
    aclTensor* inBAclTensor = nullptr;
    aclTensor* outAclTensor = nullptr;
    toAclTensor(inATensor, inAAclTensor);
    toAclTensor(inBTensor, inBAclTensor);
    toAclTensor(outTensor, outAclTensor);

    auto ret = aclnnMatmulGetWorkspaceSize(inAAclTensor, inBAclTensor, outAclTensor, cubeMathType, &workspaceSize,
                                           &executor);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "aclnnMatmulGetWorkspaceSize failed. ERROR: " << ret;
        aclDestroyTensor(inAAclTensor);
        aclDestroyTensor(inBAclTensor);
        aclDestroyTensor(outAclTensor);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
            executor = nullptr;
        }
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    ret = aclnnMatmul(deviceBuffer, workspaceSize, executor, stream);
    if (ret != ::ACL_SUCCESS) {
        aclDestroyAclOpExecutor(executor);
        ASDSIP_LOG(ERROR) << "aclnnMatmul failed. ERROR: " << ret;
        aclDestroyTensor(inAAclTensor);
        aclDestroyTensor(inBAclTensor);
        aclDestroyTensor(outAclTensor);
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    aclDestroyAclOpExecutor(executor);
    aclDestroyTensor(inAAclTensor);
    aclDestroyTensor(inBAclTensor);
    aclDestroyTensor(outAclTensor);
    ASDSIP_LOG(INFO) << "Execute MatMul success.";
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
