/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// core/blas 非法步长/leading dimension 显式拒绝负向用例（issue #385）：
// swap/ssyr2 对 incx/incy <= 0 原静默改写为 1 继续计算返回成功，ssyr/strmm/cgerc/
// cgemv_batched 对 lda/ldb/ldc <= 0 原仅打日志放行，均属非法参数显式拒绝口径（#243/#260）
// 的遗漏；修复后统一返回 ACL_ERROR_INVALID_PARAM。
// 非约定步长/leading dimension 显式拒绝负向用例（issue #484）：cal/copy/ssyr/ssyr2 的
// incx/incy 与 cgerc/ssyr/ssyr2 的 lda 均未被 tiling/kernel 消费，恒按连续内存/紧凑布局
// 计算，非约定正值（incx=2、lda=n-1 等）原静默算错；修复后按 cgemv/ctrmv 家族口径
// 统一返回 ACL_ERROR_INVALID_PARAM（docs 已声明当前约束为 1/m/n）。

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

void DestroyAclTensorWithAddr(aclTensor* tensor, void* addr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (addr != nullptr) {
        aclrtFree(addr);
    }
}

} // namespace

// swap：incx/incy <= 0 应返回 ACL_ERROR_INVALID_PARAM，不再静默改写为 1

TEST(TestBlasStrideGuard, SswapRejectNegativeIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<float> xHost(2, 1.0f);
    std::vector<float> yHost(2, 2.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {2}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSwapPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSswap(handle, 2, x, -1, y, 1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, SswapRejectNegativeIncy)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<float> xHost(2, 1.0f);
    std::vector<float> yHost(2, 2.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {2}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSwapPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSswap(handle, 2, x, 1, y, -1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// ssyr2：incx/incy/lda <= 0 应返回 ACL_ERROR_INVALID_PARAM

TEST(TestBlasStrideGuard, Ssyr2RejectNegativeIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> yHost(4, 1.0f);
    std::vector<float> aHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyr2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSsyr2(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, 4, 1.0f, x, -1, y, 1, a, 4),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, Ssyr2RejectNegativeIncy)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> yHost(4, 1.0f);
    std::vector<float> aHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyr2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSsyr2(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, 4, 1.0f, x, 1, y, -1, a, 4),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, Ssyr2RejectNegativeLda)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> yHost(4, 1.0f);
    std::vector<float> aHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyr2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSsyr2(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, 4, 1.0f, x, 1, y, 1, a, -1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// ssyr：lda <= 0 应返回 ACL_ERROR_INVALID_PARAM

TEST(TestBlasStrideGuard, SsyrRejectNegativeLda)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> aHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyrPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSsyr(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, 4, 1.0f, x, 1, a, -1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// strmm：lda/ldb/ldc <= 0 应返回 ACL_ERROR_INVALID_PARAM

TEST(TestBlasStrideGuard, StrmmRejectNegativeLda)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* a = nullptr;
    aclTensor* b = nullptr;
    aclTensor* c = nullptr;
    void* aAddr = nullptr;
    void* bAddr = nullptr;
    void* cAddr = nullptr;
    std::vector<float> aHost(16, 1.0f);
    std::vector<float> bHost(16, 1.0f);
    std::vector<float> cHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);
    ASSERT_EQ(CreateAclTensor(bHost, {4, 4}, &bAddr, aclDataType::ACL_FLOAT, &b), 0);
    ASSERT_EQ(CreateAclTensor(cHost, {4, 4}, &cAddr, aclDataType::ACL_FLOAT, &c), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeStrmmPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasStrmm(handle, asdBlasSideMode_t::ASDBLAS_SIDE_LEFT, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER,
                           asdBlasOperation_t::ASDBLAS_OP_N, asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, 4, 4, 1.0f, a,
                           -1, b, 4, c, 4),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(a, aAddr);
    DestroyAclTensorWithAddr(b, bAddr);
    DestroyAclTensorWithAddr(c, cAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, StrmmRejectNegativeLdb)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* a = nullptr;
    aclTensor* b = nullptr;
    aclTensor* c = nullptr;
    void* aAddr = nullptr;
    void* bAddr = nullptr;
    void* cAddr = nullptr;
    std::vector<float> aHost(16, 1.0f);
    std::vector<float> bHost(16, 1.0f);
    std::vector<float> cHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);
    ASSERT_EQ(CreateAclTensor(bHost, {4, 4}, &bAddr, aclDataType::ACL_FLOAT, &b), 0);
    ASSERT_EQ(CreateAclTensor(cHost, {4, 4}, &cAddr, aclDataType::ACL_FLOAT, &c), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeStrmmPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasStrmm(handle, asdBlasSideMode_t::ASDBLAS_SIDE_LEFT, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER,
                           asdBlasOperation_t::ASDBLAS_OP_N, asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, 4, 4, 1.0f, a, 4,
                           b, -1, c, 4),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(a, aAddr);
    DestroyAclTensorWithAddr(b, bAddr);
    DestroyAclTensorWithAddr(c, cAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, StrmmRejectNegativeLdc)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* a = nullptr;
    aclTensor* b = nullptr;
    aclTensor* c = nullptr;
    void* aAddr = nullptr;
    void* bAddr = nullptr;
    void* cAddr = nullptr;
    std::vector<float> aHost(16, 1.0f);
    std::vector<float> bHost(16, 1.0f);
    std::vector<float> cHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);
    ASSERT_EQ(CreateAclTensor(bHost, {4, 4}, &bAddr, aclDataType::ACL_FLOAT, &b), 0);
    ASSERT_EQ(CreateAclTensor(cHost, {4, 4}, &cAddr, aclDataType::ACL_FLOAT, &c), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeStrmmPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasStrmm(handle, asdBlasSideMode_t::ASDBLAS_SIDE_LEFT, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER,
                           asdBlasOperation_t::ASDBLAS_OP_N, asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, 4, 4, 1.0f, a, 4,
                           b, 4, c, -1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(a, aAddr);
    DestroyAclTensorWithAddr(b, bAddr);
    DestroyAclTensorWithAddr(c, cAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// cgerc：lda <= 0 应返回 ACL_ERROR_INVALID_PARAM（形状校验不覆盖 lda，可直达新增守卫）

TEST(TestBlasStrideGuard, CgercRejectNegativeLda)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> aHost(16, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_COMPLEX64, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCgercPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    std::complex<float> alpha(1.0f, 0.0f);
    EXPECT_EQ(asdBlasCgerc(handle, 4, 4, alpha, x, 1, y, 1, a, -1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// cgemv_batched：lda/incx/incy <= 0 应返回 ACL_ERROR_INVALID_PARAM

TEST(TestBlasStrideGuard, CgemvBatchedRejectNegativeLda)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* a = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* aAddr = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> aHost(16, {1.0f, 0.0f});
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_COMPLEX64, &a), 0);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCgemvBatchedPlan(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4), AsdSip::ErrorType::ACL_SUCCESS);
    std::complex<float> alpha(1.0f, 0.0f);
    std::complex<float> beta(0.0f, 0.0f);
    EXPECT_EQ(asdBlasCgemvBatched(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4, 4, alpha, a, -1, x, 1, beta, y, 1, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(a, aAddr);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, CgemvBatchedRejectNegativeIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* a = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* aAddr = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> aHost(16, {1.0f, 0.0f});
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_COMPLEX64, &a), 0);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCgemvBatchedPlan(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4), AsdSip::ErrorType::ACL_SUCCESS);
    std::complex<float> alpha(1.0f, 0.0f);
    std::complex<float> beta(0.0f, 0.0f);
    EXPECT_EQ(asdBlasCgemvBatched(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4, 4, alpha, a, 4, x, -1, beta, y, 1, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(a, aAddr);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, CgemvBatchedRejectNegativeIncy)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* a = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* aAddr = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> aHost(16, {1.0f, 0.0f});
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_COMPLEX64, &a), 0);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCgemvBatchedPlan(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4), AsdSip::ErrorType::ACL_SUCCESS);
    std::complex<float> alpha(1.0f, 0.0f);
    std::complex<float> beta(0.0f, 0.0f);
    EXPECT_EQ(asdBlasCgemvBatched(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4, 4, alpha, a, 4, x, 1, beta, y, -1, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(a, aAddr);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// 合法语义但不被支持的参数同样显式拒绝：impl 按连续内存访问并丢弃 lda/incx/incy，
// 步长 2 与 lda != m 静默成功正是 #385 要消灭的"成功返回与实际计算不对应"
TEST(TestBlasStrideGuard, CgemvBatchedRejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* a = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* aAddr = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> aHost(16, {1.0f, 0.0f});
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_COMPLEX64, &a), 0);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCgemvBatchedPlan(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4), AsdSip::ErrorType::ACL_SUCCESS);
    std::complex<float> alpha(1.0f, 0.0f);
    std::complex<float> beta(0.0f, 0.0f);
    EXPECT_EQ(asdBlasCgemvBatched(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4, 4, alpha, a, 4, x, 2, beta, y, 1, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(a, aAddr);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, CgemvBatchedRejectLdaMismatch)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* a = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* aAddr = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> aHost(16, {1.0f, 0.0f});
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_COMPLEX64, &a), 0);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCgemvBatchedPlan(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4), AsdSip::ErrorType::ACL_SUCCESS);
    std::complex<float> alpha(1.0f, 0.0f);
    std::complex<float> beta(0.0f, 0.0f);
    EXPECT_EQ(asdBlasCgemvBatched(handle, asdBlasOperation_t::ASDBLAS_OP_N, 4, 4, alpha, a, 5, x, 1, beta, y, 1, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(a, aAddr);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// ssyr2：非约定步长/leading dimension（incx=2、lda=n-1）应返回 ACL_ERROR_INVALID_PARAM
// （issue #484）：incx/incy/lda 均未被 tiling/kernel 消费，恒按连续内存/紧凑布局计算

TEST(TestBlasStrideGuard, Ssyr2RejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> yHost(4, 1.0f);
    std::vector<float> aHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyr2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSsyr2(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, 4, 1.0f, x, 2, y, 1, a, 4),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, Ssyr2RejectLdaMismatch)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> yHost(4, 1.0f);
    std::vector<float> aHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyr2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSsyr2(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, 4, 1.0f, x, 1, y, 1, a, 3),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// ssyr：非约定步长/leading dimension（incx=2、lda=n-1）应返回 ACL_ERROR_INVALID_PARAM（issue #484）

TEST(TestBlasStrideGuard, SsyrRejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> aHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyrPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSsyr(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, 4, 1.0f, x, 2, a, 4),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, SsyrRejectLdaMismatch)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> aHost(16, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 4}, &aAddr, aclDataType::ACL_FLOAT, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyrPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSsyr(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, 4, 1.0f, x, 1, a, 3),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// cgerc：非约定 leading dimension（lda=m-1）应返回 ACL_ERROR_INVALID_PARAM（issue #484）

TEST(TestBlasStrideGuard, CgercRejectLdaMismatch)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* a = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* aAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(2, {1.0f, 0.0f});
    std::vector<std::complex<float>> aHost(8, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);
    ASSERT_EQ(CreateAclTensor(aHost, {4, 2}, &aAddr, aclDataType::ACL_COMPLEX64, &a), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCgercPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCgerc(handle, 4, 2, {1.0f, 0.0f}, x, 1, y, 1, a, 3), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(a, aAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// cal（Sscal/Cscal/Csscal）：非约定步长（incx=2）应返回 ACL_ERROR_INVALID_PARAM（issue #484）：
// incx 未写入 OpParam::Cal，tiling/kernel 无步长消费，恒按连续内存计算

TEST(TestBlasStrideGuard, SscalRejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    void* xAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCalPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSscal(handle, 4, 1.0f, x, 2), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, CscalRejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    void* xAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCalPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCscal(handle, 4, {1.0f, 0.0f}, x, 2), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// copy（Scopy/Ccopy）：非约定步长（incx=2/incy=2）应返回 ACL_ERROR_INVALID_PARAM（issue #484）

TEST(TestBlasStrideGuard, ScopyRejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> yHost(4, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCopyPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasScopy(handle, 4, x, 2, y, 1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, ScopyRejectStridedIncy)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> yHost(4, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCopyPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasScopy(handle, 4, x, 1, y, 2), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasStrideGuard, CcopyRejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCopyPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCcopy(handle, 4, x, 2, y, 1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}
