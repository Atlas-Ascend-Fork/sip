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
#include "base_api.h"
#include "utils/ops_base.h"
#include "utils/common_check.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "solve.h"

using namespace Mki;

namespace AsdSip {

static AspbStatus SolveShapeCheck(const aclTensor* a, const aclTensor* b, const aclTensor* x, int64_t*& aDims,
                                  uint64_t& aDimsNum, int64_t& batch, int64_t& n, int64_t& nrhs)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(a, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(b, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(x, aclDataType::ACL_FLOAT, ret);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(a, &aDims, &aDimsNum), "solve: aclGetStorageShape(A)");
    if (aDimsNum != 3 || aDims[1] != aDims[2]) {
        delete[] aDims;
        ASDSIP_LOG(ERROR) << "solve: A must be [batch, n, n], got dimsNum " << aDimsNum;
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    batch = aDims[0];
    n = aDims[1];

    int64_t* bDims = nullptr;
    uint64_t bDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(b, &bDims, &bDimsNum), "solve: aclGetStorageShape(B)");
    if (bDimsNum != 3 || bDims[0] != batch || bDims[1] != n) {
        delete[] bDims;
        delete[] aDims;
        aDims = nullptr;
        ASDSIP_LOG(ERROR) << "solve: B must be [batch, n, nrhs] with same batch/n as A (broadcast 请调用方先展开)";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    nrhs = bDims[2];

    int64_t* xDims = nullptr;
    uint64_t xDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(x, &xDims, &xDimsNum), "solve: aclGetStorageShape(X)");
    bool xOk = (xDimsNum == 3 && xDims[0] == batch && xDims[1] == n && xDims[2] == nrhs);
    delete[] xDims;
    if (!xOk) {
        delete[] bDims;
        delete[] aDims;
        aDims = nullptr;
        ASDSIP_LOG(ERROR) << "solve: X must be [batch, n, nrhs], same as B";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    delete[] bDims;
    return ErrorType::ACL_SUCCESS;
}

// 外部 aclTensor 接口: 求解 A·X = B (fp32/ND, n∈[2,64], B 需为满 batch 形态)
AspbStatus solve(const aclTensor* a, const aclTensor* b, aclTensor* x, void* stream, void* workspace)
{
    int64_t* aDims = nullptr;
    uint64_t aDimsNum = 0;
    int64_t batch = 0;
    int64_t n = 0;
    int64_t nrhs = 0;
    auto ret = SolveShapeCheck(a, b, x, aDims, aDimsNum, batch, n, nrhs);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }
    if (aDims != nullptr) {
        delete[] aDims;
        aDims = nullptr;
    }

    OpDesc opDesc;
    opDesc.opName = "SolveOperation";
    AsdSip::OpParam::Solve param;
    param.batch = batch;
    param.n = n;
    param.nrhs = nrhs;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor*> inTensors = {const_cast<aclTensor*>(a), const_cast<aclTensor*>(b)};
    SVector<aclTensor*> outTensors = {x};

    Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, static_cast<uint8_t*>(workspace));
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_LOG(INFO) << "Execute solve success.";
    return ErrorType::ACL_SUCCESS;
}

AspbStatus solveGetWorkspaceSize(size_t& size)
{
    size = 0; // 无 device scratch
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
