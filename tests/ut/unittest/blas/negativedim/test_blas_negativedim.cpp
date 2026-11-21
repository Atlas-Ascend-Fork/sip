/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// core/blas 0 维（标量）张量负向用例（issue #343）：host 对 aclGetStorageShape 返回的
// storageDims 判空后应返回明确错误码，不再空指针解引用崩溃；
// csrot n 上界负向用例（issue #344）：n > INT32_MAX/2 应被拒绝而非静默截断；
// 向量 BLAS 族 n 上界正/负向用例（issue #424）：复数 2n 载荷入口 INT32_MAX/2 边界放行、
// 实数计数入口 INT32_MAX 边界放行，越界均拒绝。
// 实数入口 Sasum/Snrm2/Scopy/Sswap 上限统一与 sdot/isamax 对齐（issue #447）；
// 复数 2n 载荷入口 Ccopy/Cswap 上限收紧至 INT32_MAX/2（issue #444）；
// Sasum/Cswap 报错文案与实际函数名一一对应（issue #448）；
// Cal/Caxpy 族 4 入口上限同步收紧：Sscal 实数计数、Csscal/Cscal/Caxpy 复数 2n 载荷（issue #451）；
// Cswap 边界断言恢复（!367 误删）+ Caxpy 越界期望值改透传口径 INVALID_PARAM 防校验回退（issue #455）。

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

// 0 维（标量）aclTensor：shape/dims 均为空，aclGetStorageShape 对其返回 dimsNum=0 且 dims=nullptr
int CreateScalarAclTensor(aclDataType dtype, size_t byteSize, aclTensor** tensor, void** addr)
{
    auto ret = aclrtMalloc(addr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ::ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    *tensor = aclCreateTensor(nullptr, 0, dtype, nullptr, 0, aclFormat::ACL_FORMAT_ND, nullptr, 0, *addr);
    return *tensor != nullptr ? 0 : -1;
}

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

// 以下用例统一断言：0 维输入返回 ACL_ERROR_OP_INPUT_NOT_MATCH，不崩溃

TEST(TestBlasZeroDimGuard, Sasum)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &x, &xAddr), 0);
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeAsumPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSasum(handle, 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Scasum)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeAsumPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasScasum(handle, 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// asum result 侧 0 维（issue #343 补充覆盖 result 解引用点）
TEST(TestBlasZeroDimGuard, SasumResult)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<float> xHost(1, 1.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &result, &resultAddr), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeAsumPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSasum(handle, 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Sscal)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    void* xAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &x, &xAddr), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCalPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSscal(handle, 1, 2.0f, x, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Cscal)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    void* xAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCalPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    std::complex<float> alpha(2.0f, 1.0f);
    EXPECT_EQ(asdBlasCscal(handle, 1, alpha, x, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Csscal)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    void* xAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCalPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCsscal(handle, 1, 2.0f, x, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Isamax)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &x, &xAddr), 0);
    std::vector<int32_t> resultHost(1, 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_INT32, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeIamaxPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasIsamax(handle, 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Icamax)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);
    std::vector<int32_t> resultHost(1, 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_INT32, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeIamaxPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasIcamax(handle, 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Csrot)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);
    std::vector<std::complex<float>> yHost(2, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeRotPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCsrot(handle, 1, x, 1, y, 1, 1.0f, 0.0f), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Sswap)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &x, &xAddr), 0);
    std::vector<float> yHost(2, 1.0f);
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSwapPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSswap(handle, 1, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Cswap)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);
    std::vector<std::complex<float>> yHost(2, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSwapPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCswap(handle, 1, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Scopy)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &x, &xAddr), 0);
    std::vector<float> yHost(2, 1.0f);
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCopyPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasScopy(handle, 1, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Ccopy)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);
    std::vector<std::complex<float>> yHost(2, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCopyPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCcopy(handle, 1, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Sdot)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &x, &xAddr), 0);
    std::vector<float> yHost(2, 1.0f);
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeDotPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSdot(handle, 1, x, 1, y, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Cdotu)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);
    std::vector<std::complex<float>> yHost(2, {1.0f, 0.0f});
    std::vector<std::complex<float>> resultHost(1, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_COMPLEX64, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeDotPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCdotu(handle, 1, x, 1, y, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Cdotc)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);
    std::vector<std::complex<float>> yHost(2, {1.0f, 0.0f});
    std::vector<std::complex<float>> resultHost(1, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(yHost, {2}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_COMPLEX64, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeDotPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCdotc(handle, 1, x, 1, y, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Snrm2)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &x, &xAddr), 0);
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeNrm2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSnrm2(handle, 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimGuard, Scnrm2)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &x, &xAddr), 0);
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeNrm2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasScnrm2(handle, 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// csrot n 上界（issue #344）：n > INT32_MAX/2 应返回 ACL_ERROR_INVALID_PARAM 而非静默截断；
// 小张量即可触达（host 校验在 shape 一致性检查之前），无需构造超大张量
TEST(TestBlasCsrotNBound, RejectOverInt32Half)
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
    EXPECT_EQ(asdBlasCsrot(handle, INT32_MAX / 2 + 1, x, 1, y, 1, 1.0f, 0.0f),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// 向量 BLAS 族 n 上界用例（issue #424，按载荷语义分档，检视意见 !335 修正）：
// 复数 2n 载荷入口（cdotu/cdotc/scasum/scnrm2）上限 INT32_MAX/2（2n 不越 uint32 承载
// 与 int32 元素偏移正域，对齐 rot #344 口径）；实数计数入口（sdot/isamax/icamax）上限
// INT32_MAX（=UINT32_MAX/2，uint32 承载不回绕且元素偏移不越 int32 正域，不适用 rot
// #344 的 2n 特例）。越界均显式拒绝；边界值须放行——小张量即可触达（host 校验在
// shape 一致性检查之前），放行证据为后续 shape 校验的 OP_INPUT_NOT_MATCH
TEST(TestBlasVectorNBound, Sdot)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> yHost(4, 1.0f);
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_FLOAT, &y), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeDotPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSdot(handle, INT32_MAX, x, 1, y, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasSdot(handle, static_cast<int64_t>(INT32_MAX) + 1, x, 1, y, 1, result),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Cdotu)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> resultHost(1, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_COMPLEX64, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeDotPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCdotu(handle, INT32_MAX / 2, x, 1, y, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasCdotu(handle, INT32_MAX / 2 + 1, x, 1, y, 1, result), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Cdotc)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> yHost(4, {1.0f, 0.0f});
    std::vector<std::complex<float>> resultHost(1, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(yHost, {4}, &yAddr, aclDataType::ACL_COMPLEX64, &y), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_COMPLEX64, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeDotPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCdotc(handle, INT32_MAX / 2, x, 1, y, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasCdotc(handle, INT32_MAX / 2 + 1, x, 1, y, 1, result), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Scasum)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeAsumPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasScasum(handle, INT32_MAX / 2, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasScasum(handle, INT32_MAX / 2 + 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Scnrm2)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeNrm2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasScnrm2(handle, INT32_MAX / 2, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasScnrm2(handle, INT32_MAX / 2 + 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Isamax)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<int32_t> resultHost(1, 0);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_INT32, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeIamaxPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasIsamax(handle, INT32_MAX, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasIsamax(handle, static_cast<int64_t>(INT32_MAX) + 1, x, 1, result),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Icamax)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    std::vector<int32_t> resultHost(1, 0);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_INT32, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeIamaxPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasIcamax(handle, INT32_MAX, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasIcamax(handle, static_cast<int64_t>(INT32_MAX) + 1, x, 1, result),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// 实数计数入口上限统一收紧至 INT32_MAX（issue #447）：边界值放行（后续 shape 校验
// 的 OP_INPUT_NOT_MATCH 为放行证据），越界显式拒绝
TEST(TestBlasVectorNBound, Sasum)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeAsumPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSasum(handle, INT32_MAX, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasSasum(handle, static_cast<int64_t>(INT32_MAX) + 1, x, 1, result),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Snrm2)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    std::vector<float> resultHost(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_FLOAT, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeNrm2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSnrm2(handle, INT32_MAX, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasSnrm2(handle, static_cast<int64_t>(INT32_MAX) + 1, x, 1, result),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// copy impl 的张量数量校验错误码为 OP_OUTPUT_NOT_MATCH（既有口径），作放行证据
TEST(TestBlasVectorNBound, Scopy)
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
    EXPECT_EQ(asdBlasScopy(handle, INT32_MAX, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_OP_OUTPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasScopy(handle, static_cast<int64_t>(INT32_MAX) + 1, x, 1, y, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Sswap)
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
    ASSERT_EQ(asdBlasMakeSwapPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSswap(handle, INT32_MAX, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasSswap(handle, static_cast<int64_t>(INT32_MAX) + 1, x, 1, y, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// 复数 2n 载荷入口上限收紧至 INT32_MAX/2（issue #444）：2n 不越 uint32 承载与
// int32 元素偏移正域，边界值放行（后续 shape 校验错误码为放行证据），越界显式拒绝
TEST(TestBlasVectorNBound, Ccopy)
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
    EXPECT_EQ(asdBlasCcopy(handle, INT32_MAX / 2, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_OP_OUTPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasCcopy(handle, INT32_MAX / 2 + 1, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Cswap)
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
    ASSERT_EQ(asdBlasMakeSwapPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCswap(handle, INT32_MAX / 2, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasCswap(handle, INT32_MAX / 2 + 1, x, 1, y, 1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// Cal/Caxpy 族 4 入口上限同步收紧（issue #451，对齐 64cadc0 已合口径）：
// 实数计数入口 Sscal 收紧至 INT32_MAX，2n 载荷入口 Csscal/Cscal/Caxpy 收紧至 INT32_MAX/2，
// 边界值放行（后续 shape 校验错误码为放行证据），越界显式拒绝
TEST(TestBlasVectorNBound, Sscal)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    void* xAddr = nullptr;
    std::vector<float> xHost(4, 1.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCalPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSscal(handle, INT32_MAX, 1.0f, x, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasSscal(handle, static_cast<int64_t>(INT32_MAX) + 1, 1.0f, x, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Csscal)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    void* xAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCalPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCsscal(handle, INT32_MAX / 2, 1.0f, x, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasCsscal(handle, INT32_MAX / 2 + 1, 1.0f, x, 1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Cscal)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    void* xAddr = nullptr;
    std::vector<std::complex<float>> xHost(4, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {4}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCalPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCscal(handle, INT32_MAX / 2, {1.0f, 0.0f}, x, 1), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    EXPECT_EQ(asdBlasCscal(handle, INT32_MAX / 2 + 1, {1.0f, 0.0f}, x, 1), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasVectorNBound, Caxpy)
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
    ASSERT_EQ(asdBlasMakeCaxpyPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasCaxpy(handle, INT32_MAX / 2, {1.0f, 0.0f}, x, 1, y, 1),
              AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    // 越界拒绝（issue #455）：外层包装透传内层实际错误码，n 上界校验失败为 INVALID_PARAM，
    // 与边界放行（触达 NUM_NOT_MATCH 为 OP_INPUT_NOT_MATCH）可区分，防校验回退
    EXPECT_EQ(asdBlasCaxpy(handle, INT32_MAX / 2 + 1, {1.0f, 0.0f}, x, 1, y, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(y, yAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}
