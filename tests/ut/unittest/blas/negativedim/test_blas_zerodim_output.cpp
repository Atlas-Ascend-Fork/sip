/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// core/blas 输出侧（result/y）0 维（标量）张量负向用例（issue #383）：
// #343 的输出侧判空被写进了解引用之后的分支（对空指针场景不可达）或缺失（rot y 侧），
// 修复后判空必须位于解引用之前，0 维输出统一返回 ACL_ERROR_OP_INPUT_NOT_MATCH，不崩溃。
// x 侧 0 维覆盖见 test_blas_negativedim.cpp（issue #343），本文件只补输出侧解引用点。

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

// 以下用例统一断言：0 维输出（result/y 侧）返回 ACL_ERROR_OP_INPUT_NOT_MATCH，不崩溃

TEST(TestBlasZeroDimOutputGuard, IsamaxResult)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<float> xHost(1, 1.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_INT32, sizeof(int32_t), &result, &resultAddr), 0);

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

TEST(TestBlasZeroDimOutputGuard, IcamaxResult)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost(1, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_INT32, sizeof(int32_t), &result, &resultAddr), 0);

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

TEST(TestBlasZeroDimOutputGuard, Snrm2Result)
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
    ASSERT_EQ(asdBlasMakeNrm2Plan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasSnrm2(handle, 1, x, 1, result), AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasZeroDimOutputGuard, Scnrm2Result)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost(1, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &result, &resultAddr), 0);

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

TEST(TestBlasZeroDimOutputGuard, ScopyY)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<float> xHost(1, 1.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &y, &yAddr), 0);

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

TEST(TestBlasZeroDimOutputGuard, CcopyY)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> xHost(1, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &y, &yAddr), 0);

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

TEST(TestBlasZeroDimOutputGuard, SswapY)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<float> xHost(1, 1.0f);
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &y, &yAddr), 0);

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

TEST(TestBlasZeroDimOutputGuard, CswapY)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> xHost(1, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &y, &yAddr), 0);

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

TEST(TestBlasZeroDimOutputGuard, CsrotY)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    void* xAddr = nullptr;
    void* yAddr = nullptr;
    std::vector<std::complex<float>> xHost(1, {1.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(xHost, {1}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, 2 * sizeof(float), &y, &yAddr), 0);

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
