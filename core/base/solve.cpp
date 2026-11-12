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

static AspbStatus SolveShapeCheck(const aclTensor* a, const aclTensor* b, const aclTensor* x, int64_t& batch,
                                  int64_t& n, int64_t& nrhs)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(a, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(b, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(x, aclDataType::ACL_FLOAT, ret);

    int64_t* aDims = nullptr;
    uint64_t aDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(a, &aDims, &aDimsNum), "solve: aclGetStorageShape(A)");
    const bool aOk = (aDimsNum == 3 && aDims[1] == aDims[2]);
    if (aOk) {
        batch = aDims[0];
        n = aDims[1];
    }
    delete[] aDims;
    ASDSIP_CHECK(aOk, "solve: A must be [batch, n, n]", return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    ASDSIP_CHECK(batch >= 1, "solve: batch must be >= 1", return ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_CHECK(n >= 2 && n <= 256, "solve: n must be in [2, 256]", return ErrorType::ACL_ERROR_INVALID_PARAM);

    int64_t* bDims = nullptr;
    uint64_t bDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(b, &bDims, &bDimsNum), "solve: aclGetStorageShape(B)");
    // B 的 batch 允许为 1 (广播形态, kernel 寻址时模掉 batch 偏移) 或与 A 一致
    const bool bOk = (bDimsNum == 3 && (bDims[0] == batch || bDims[0] == 1) && bDims[1] == n);
    if (bOk) {
        nrhs = bDims[2];
    }
    delete[] bDims;
    ASDSIP_CHECK(bOk, "solve: B must be [batch, n, nrhs] with same batch/n as A (B batch=1 broadcast is supported)",
                 return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    ASDSIP_CHECK(nrhs >= 1, "solve: cols(nrhs) must be >= 1", return ErrorType::ACL_ERROR_INVALID_PARAM);

    int64_t* xDims = nullptr;
    uint64_t xDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(x, &xDims, &xDimsNum), "solve: aclGetStorageShape(X)");
    // X 的 batch 跟随 A (B 为 [1, n, nrhs] 广播形态时, 输出仍为 [batch, n, nrhs])
    const bool xOk = (xDimsNum == 3 && xDims[0] == batch && xDims[1] == n && xDims[2] == nrhs);
    delete[] xDims;
    ASDSIP_CHECK(xOk, "solve: X must be [batch, n, nrhs] with batch of A",
                 return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

// 求解 A·X = B (fp32/ND, n∈[2,256], B 可为满 batch 或 batch=1 广播形态)
AspbStatus solve(const aclTensor* a, const aclTensor* b, aclTensor* x, void* stream, void* workspace)
{
    int64_t batch = 0;
    int64_t n = 0;
    int64_t nrhs = 0;
    auto ret = SolveShapeCheck(a, b, x, batch, n, nrhs);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
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
