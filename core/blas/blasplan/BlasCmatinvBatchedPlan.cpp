/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasCmatinvBatchedPlan.h"
#include <securec.h>
#include "mki/launch_param.h"
#include "utils/assert.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

static constexpr int ELEMENT_NUM_PER_BLOCK = 8;
static constexpr int ELEMENT_NUM_PER_REPEAT = 64;
static constexpr uint32_t BASE_COMPLEX64_NUM = 8192; // 64kb / 8b
static constexpr uint32_t MAX_COMPUTE_NUM = 128;     // max compute num per repeat

namespace AsdSip {
BlasCmatinvBatchedPlan::BlasCmatinvBatchedPlan(int64_t n) : BlasPlan(), n{n}
{
    uniMatReal.data = nullptr;
    uniMatReal.hostData = nullptr;
    uniMatImag.data = nullptr;
    uniMatImag.hostData = nullptr;
    offsetTensor.data = nullptr;
    offsetTensor.hostData = nullptr;
};

AsdSip::AspbStatus BlasCmatinvBatchedPlan::CreateTensor()
{
    uint32_t N = static_cast<uint32_t>(n);
    uint32_t alignedN = (N + (ELEMENT_NUM_PER_BLOCK - 1)) / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;

    uint32_t computeNumPerRepeat = BASE_COMPLEX64_NUM / (N * alignedN);
    computeNumPerRepeat = std::min(computeNumPerRepeat, MAX_COMPUTE_NUM);
    computeNumPerRepeat = computeNumPerRepeat / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;

    std::vector<float> uniBatchRealData(N * alignedN * computeNumPerRepeat, 0.0f);
    std::vector<float> uniMatRealData(N * alignedN, 0.0f);
    std::vector<float> uniBatchImagData(N * alignedN * computeNumPerRepeat, 0.0f);
    std::vector<float> uniMatImagData(N * alignedN, 0.0f);

    for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++) {
        for (uint32_t colIdx = 0; colIdx < alignedN; colIdx++) {
            if (rowIdx == colIdx) {
                uniMatRealData[alignedN * rowIdx + colIdx] = 1.0f;
            } else {
                uniMatRealData[alignedN * rowIdx + colIdx] = 0.0f;
            }
            uniMatImagData[alignedN * rowIdx + colIdx] = 0.0f;
        }
    }

    errno_t rc;
    for (uint32_t i = 0; i < computeNumPerRepeat; i++) {
        rc = memcpy_s(static_cast<float*>(uniBatchRealData.data() + N * alignedN * i), N * alignedN * sizeof(float),
                      uniMatRealData.data(), N * alignedN * sizeof(float));
        ASDSIP_ECHECK(rc == EOK, "Repeat uniMatRealData by memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        rc = memcpy_s(static_cast<float*>(uniBatchImagData.data() + N * alignedN * i), N * alignedN * sizeof(float),
                      uniMatImagData.data(), N * alignedN * sizeof(float));
        ASDSIP_ECHECK(rc == EOK, "Repeat uniMatImagData by memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    }

    uint32_t offsetSize = N * alignedN * computeNumPerRepeat * 2;
    offsetSize = (offsetSize + (ELEMENT_NUM_PER_REPEAT - 1)) / ELEMENT_NUM_PER_REPEAT * ELEMENT_NUM_PER_REPEAT;
    std::vector<uint32_t> offsetData(offsetSize, 0);

    uint32_t k = 0;
    uint32_t realBase = 0;
    uint32_t imagBase = 32 * 1024;
    for (uint32_t batchIdx = 0; batchIdx < computeNumPerRepeat; batchIdx++) {
        for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++) {
            for (uint32_t colIdx = 0; colIdx < N; colIdx++) {
                offsetData[k++] = realBase + sizeof(uint32_t) * (batchIdx * alignedN * N + alignedN * rowIdx + colIdx);
                offsetData[k++] = imagBase + sizeof(uint32_t) * (batchIdx * alignedN * N + alignedN * rowIdx + colIdx);
            }
        }
    }

    uniMatReal.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {computeNumPerRepeat, N, alignedN}, {}, 0};
    uniMatReal.hostData = uniBatchRealData.data();
    uniMatReal.dataSize = N * alignedN * computeNumPerRepeat * sizeof(float);
    if (!MallocTensorInDevice(uniMatReal).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCmatinvBatchedPlan uniMatReal malloc failed";
        FreeTensor();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (!toAclTensor(uniMatReal, uniMatAclReal).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCmatinvBatchedPlan uniMatReal toAclTensor failed";
        FreeTensor();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    uniMatImag.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {computeNumPerRepeat, N, alignedN}, {}, 0};
    uniMatImag.hostData = uniBatchImagData.data();
    uniMatImag.dataSize = N * alignedN * computeNumPerRepeat * sizeof(float);
    if (!MallocTensorInDevice(uniMatImag).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCmatinvBatchedPlan uniMatImag malloc failed";
        FreeTensor();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (!toAclTensor(uniMatImag, uniMatAclImag).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCmatinvBatchedPlan uniMatImag toAclTensor failed";
        FreeTensor();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    offsetTensor.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {offsetSize}, {}, 0};
    offsetTensor.hostData = offsetData.data();
    offsetTensor.dataSize = offsetSize * sizeof(uint32_t);
    if (!MallocTensorInDevice(offsetTensor).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCmatinvBatchedPlan offsetTensor malloc failed";
        FreeTensor();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (!toAclTensor(offsetTensor, offsetAclTensor).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCmatinvBatchedPlan offsetTensor toAclTensor failed";
        FreeTensor();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasCmatinvBatchedPlan::FreeTensor()
{
    if (uniMatReal.data != nullptr) {
        FreeTensorInDevice(uniMatReal);
        uniMatReal.data = nullptr;
    }
    if (uniMatReal.hostData != nullptr) {
        uniMatReal.hostData = nullptr;
    }

    if (uniMatImag.data != nullptr) {
        FreeTensorInDevice(uniMatImag);
        uniMatImag.data = nullptr;
    }
    if (uniMatImag.hostData != nullptr) {
        uniMatImag.hostData = nullptr;
    }

    if (offsetTensor.data != nullptr) {
        FreeTensorInDevice(offsetTensor);
        offsetTensor.data = nullptr;
    }
    if (offsetTensor.hostData != nullptr) {
        offsetTensor.hostData = nullptr;
    }

    aclDestroyTensor(uniMatAclReal);
    aclDestroyTensor(uniMatAclImag);
    aclDestroyTensor(offsetAclTensor);

    return ErrorType::ACL_SUCCESS;
}

BlasCmatinvBatchedPlan::~BlasCmatinvBatchedPlan() { BlasPlan::DestroyPlanData(); }
} // namespace AsdSip
