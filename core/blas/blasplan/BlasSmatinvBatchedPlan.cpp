/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasSmatinvBatchedPlan.h"
#include <securec.h>
#include "mki/launch_param.h"
#include "utils/assert.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

static constexpr int ELEMENT_NUM_PER_BLOCK = 8; // fp32 elems per 32B block
static constexpr int ELEMENT_NUM_PER_REPEAT = 64;
static constexpr uint32_t BASE_FP32_NUM = 8192;  // 32KB / 4B
static constexpr uint32_t MAX_COMPUTE_NUM = 128; // max compute num per repeat

namespace AsdSip {
BlasSmatinvBatchedPlan::BlasSmatinvBatchedPlan(int64_t n) : BlasPlan(), n{n}
{
    uniMatReal.data = nullptr;
    uniMatReal.hostData = nullptr;
};

AsdSip::AspbStatus BlasSmatinvBatchedPlan::CreateTensor()
{
    /*
        Build the pre-computed identity matrix batch used by the kernel.
        Rows are padded to 32B (8 fp32 elems) alignment to match the kernel UB layout.
    */
    uint32_t N = static_cast<uint32_t>(n);
    uint32_t alignedN = (N + (ELEMENT_NUM_PER_BLOCK - 1)) / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;

    uint32_t computeNumPerRepeat = BASE_FP32_NUM / (N * alignedN);
    computeNumPerRepeat = std::min(computeNumPerRepeat, MAX_COMPUTE_NUM);
    computeNumPerRepeat = computeNumPerRepeat / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;

    std::vector<float> uniBatchRealData(N * alignedN * computeNumPerRepeat, 0.0f);
    std::vector<float> uniMatRealData(N * alignedN, 0.0f);

    for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++) {
        for (uint32_t colIdx = 0; colIdx < alignedN; colIdx++) {
            uniMatRealData[alignedN * rowIdx + colIdx] = (rowIdx == colIdx) ? 1.0f : 0.0f;
        }
    }

    errno_t rc = EOK;
    for (uint32_t i = 0; i < computeNumPerRepeat; i++) {
        rc = memcpy_s(static_cast<float*>(uniBatchRealData.data() + N * alignedN * i), N * alignedN * sizeof(float),
                      uniMatRealData.data(), N * alignedN * sizeof(float));
        ASDSIP_ECHECK(rc == EOK, "Repeat uniMatRealData by memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    }

    uniMatReal.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {computeNumPerRepeat, N, alignedN}, {}, 0};
    uniMatReal.hostData = uniBatchRealData.data();
    uniMatReal.dataSize = N * alignedN * computeNumPerRepeat * sizeof(float);
    MallocTensorInDevice(uniMatReal);
    toAclTensor(uniMatReal, uniMatAclReal);

    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasSmatinvBatchedPlan::FreeTensor()
{
    if (uniMatReal.data != nullptr) {
        FreeTensorInDevice(uniMatReal);
        uniMatReal.data = nullptr;
    }
    if (uniMatReal.hostData != nullptr) {
        uniMatReal.hostData = nullptr;
    }

    aclDestroyTensor(uniMatAclReal);

    return ErrorType::ACL_SUCCESS;
}

BlasSmatinvBatchedPlan::~BlasSmatinvBatchedPlan() { BlasPlan::DestroyPlanData(); }
} // namespace AsdSip
