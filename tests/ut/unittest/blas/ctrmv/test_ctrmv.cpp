/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// asdBlasCtrmv 枚举参数负向用例（issue #345）：side 之外的 uplo/trans/diag 超出枚举域时
// 应返回 ACL_ERROR_INVALID_PARAM，而非静默映射为错误计算模式（trans=-1 原样透传 tiling，
// 内核对非 1/2 回落 OP_N，结果错误但返回成功）

#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "blas_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;

namespace {

constexpr int64_t N = 4;

// 建句柄/plan/张量并执行一次非法枚举调用，返回接口错误码
AspbStatus RunCtrmvWithEnum(asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* aclA = nullptr;
    aclTensor* aclX = nullptr;
    void* aAddr = nullptr;
    void* xAddr = nullptr;
    std::vector<std::complex<float>> aHost(N * N, {1.0f, 0.0f});
    std::vector<std::complex<float>> xHost(N, {1.0f, 0.0f});
    int rc = CreateAclTensor(aHost, {N, N}, &aAddr, aclDataType::ACL_COMPLEX64, &aclA);
    if (rc != 0) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    rc = CreateAclTensor(xHost, {N}, &xAddr, aclDataType::ACL_COMPLEX64, &aclX);
    if (rc != 0) {
        aclDestroyTensor(aclA);
        aclrtFree(aAddr);
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    asdBlasHandle handle;
    AspbStatus st = asdBlasCreate(handle);
    if (st == AsdSip::ErrorType::ACL_SUCCESS) {
        st = asdBlasMakeCtrmvPlan(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, N);
    }
    if (st == AsdSip::ErrorType::ACL_SUCCESS) {
        st = asdBlasCtrmv(handle, uplo, trans, diag, N, aclA, N, aclX, 1);
    }

    asdBlasDestroy(handle);
    aclDestroyTensor(aclA);
    aclDestroyTensor(aclX);
    aclrtFree(aAddr);
    aclrtFree(xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
    return st;
}

TEST(TestBlasCtrmvEnumGuard, RejectInvalidTrans)
{
    EXPECT_EQ(RunCtrmvWithEnum(asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, static_cast<asdBlasOperation_t>(42),
                               asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
}

TEST(TestBlasCtrmvEnumGuard, RejectInvalidUplo)
{
    EXPECT_EQ(RunCtrmvWithEnum(static_cast<asdBlasFillMode_t>(42), asdBlasOperation_t::ASDBLAS_OP_N,
                               asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
}

TEST(TestBlasCtrmvEnumGuard, RejectInvalidDiag)
{
    EXPECT_EQ(RunCtrmvWithEnum(asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, asdBlasOperation_t::ASDBLAS_OP_N,
                               static_cast<asdBlasDiagType_t>(42)),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
}

} // namespace
