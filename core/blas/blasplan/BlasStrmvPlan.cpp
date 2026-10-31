/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "include/blasplan/BlasStrmvPlan.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"

constexpr int64_t M0_DEFAULT_SIZE = 128;

using namespace Mki;

namespace AsdSip {
constexpr int64_t BYTES_1k = 1024;
BlasStrmvPlan::BlasStrmvPlan(asdBlasFillMode_t uplo, asdBlasOperation_t trans, int64_t n)
    : BlasPlan(), uplo{uplo}, trans{trans}, n{n}
{
    maskTensor.data = nullptr;
    maskTensor.hostData = nullptr;
};

AsdSip::AspbStatus BlasStrmvPlan::CreateTensor() { return SetMaskTensor(); }

AsdSip::AspbStatus BlasStrmvPlan::SetMaskTensor()
{
    int64_t m0 = M0_DEFAULT_SIZE;

    float* uploMatrixHost = nullptr;
    try {
        uploMatrixHost = new float[m0 * m0];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasStrmvPlan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    float ele = (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER) ? 1 : 0;
    for (int64_t i = 0; i < m0; ++i) {
        for (int64_t j = 0; j < m0; ++j) {
            if (j < i) {
                *(uploMatrixHost + i * m0 + j) = ele;
            } else if (j == i) {
                *(uploMatrixHost + i * m0 + j) = 1;
            } else {
                *(uploMatrixHost + i * m0 + j) = 1 - ele;
            }
        }
    }
    maskTensor.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m0, m0}, {}, 0};
    maskTensor.hostData = uploMatrixHost;
    maskTensor.dataSize = sizeof(float) * m0 * m0;
    if (!MallocTensorInDevice(maskTensor).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasStrmvPlan maskTensor malloc failed: ";
        delete[] uploMatrixHost;
        uploMatrixHost = nullptr;
        maskTensor.hostData = nullptr;
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    uploMatrixHost = nullptr;

    toAclTensor(maskTensor, maskAclTensor);
    if (maskAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "BlasStrmvPlan convert to aclTensor fail.";
        FreeTensor();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasStrmvPlan::FreeTensor()
{
    if (maskTensor.data != nullptr) {
        FreeTensorInDevice(maskTensor);
        maskTensor.data = nullptr;
    }
    if (maskTensor.hostData != nullptr) {
        delete[] static_cast<float*>(maskTensor.hostData);
        maskTensor.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

BlasStrmvPlan::~BlasStrmvPlan() { BlasPlan::DestroyPlanData(); }

int64_t BlasStrmvPlan::GetWorkspaceSize() { return n * sizeof(float) + BYTES_1k; }
} // namespace AsdSip
