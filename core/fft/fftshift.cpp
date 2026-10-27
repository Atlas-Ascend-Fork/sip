/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <optional>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "utils/assert.h"
#include "log/log.h"
#include "utils/common_check.h"
#include "fft_api.h"
#include "utils/ops_base.h"
#include "params/fftshift.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {

// 当前版本仅支持 2D 切片（row x col）的 fftshift，其他维度支持待后续版本补齐。
AspbStatus FftShiftDtypeCheck(const aclTensor* inTensor, const aclTensor* outTensor)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(inTensor, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(outTensor, aclDataType::ACL_COMPLEX64, ret);

    return ErrorType::ACL_SUCCESS;
}

AspbStatus FftShiftShapeCheck(int64_t row, int64_t col, int64_t batch, int64_t dist, const aclTensor* inTensor,
                              const aclTensor* outTensor)
{
    int64_t sliceSize = row * col;
    int64_t requiredInNum = (batch - 1) * dist + sliceSize;
    int64_t requiredOutNum = batch * sliceSize;

    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(inTensor, &storageDims, &storageDimsNum),
                                 "asdFftShift: aclGetStorageShape of inTensor");
    std::optional<int64_t> inNumOpt = GetTensorNum(storageDimsNum, storageDims);
    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(outTensor, &storageDims, &storageDimsNum),
                                 "asdFftShift: aclGetStorageShape of outTensor");
    std::optional<int64_t> outNumOpt = GetTensorNum(storageDimsNum, storageDims);
    delete[] storageDims;
    storageDims = nullptr;

    if (!inNumOpt.has_value() || inNumOpt.value() < requiredInNum) {
        ASDSIP_LOG(ERROR) << "asdFftShift get wrong inTensor num. In element num is " << inNumOpt.value_or(-1)
                          << " , but at least " << requiredInNum << " elements are required.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    if (!outNumOpt.has_value() || outNumOpt.value() < requiredOutNum) {
        ASDSIP_LOG(ERROR) << "asdFftShift get wrong outTensor num. Out element num is " << outNumOpt.value_or(-1)
                          << " , but at least " << requiredOutNum << " elements are required.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip

AsdSip::AspbStatus AsdSip::asdFftShift(int64_t row, int64_t col, int64_t batch, int64_t dist, const aclTensor* input,
                                       aclTensor* output, void* stream, void* workspace)
{
    if (row <= 0 || col <= 0 || batch <= 0) {
        ASDSIP_LOG(ERROR) << "asdFftShift requires positive row/col/batch, got row:" << row << " , col:" << col
                          << " , batch:" << batch;
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    int64_t sliceSize = row * col;
    if (dist <= 0) {
        dist = sliceSize;
    }

    auto ret = FftShiftDtypeCheck(input, output);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }
    ret = FftShiftShapeCheck(row, col, batch, dist, input, output);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }

    OpDesc opDesc;
    opDesc.opName = "FftShiftOperation";
    AsdSip::OpParam::FftShift param;
    param.row = row;
    param.col = col;
    param.batch = batch;
    param.dist = dist;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor*> inTensors = {const_cast<aclTensor*>(input)};
    SVector<aclTensor*> outTensors = {output};

    Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, static_cast<uint8_t*>(workspace));
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_LOG(INFO) << "Execute asdFftShift success.";
    return ErrorType::ACL_SUCCESS;
}
