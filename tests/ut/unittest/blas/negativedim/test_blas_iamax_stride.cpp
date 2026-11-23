/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// iamax 步长约定负向用例（issue #506）：asdBlasIsamax/asdBlasIcamax 入口原仅校验 incx > 0，
// 非约定正值步长放行后 tiling/kernel 两侧均不消费、恒按连续布局计算，结果静默错误；
// 修复后入口对 incx != 1 显式拒绝返回 ACL_ERROR_INVALID_PARAM（docs 已声明当前约束为 1），
// tiling/kernel 两侧零消费的 incx 字段同步清理。
// iamax 功能正向用例：n=8 incx=1 实数/复数 argmax 索引与 golden 一致（1-based），
// 覆盖 tiling 布局调整后的端到端执行路径。

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

TEST(TestBlasIamaxStrideGuard, IsamaxRejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<float> xHost = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f, 4.5f, -1.5f, 0.25f};
    std::vector<int32_t> resultHost(1, 0);
    ASSERT_EQ(CreateAclTensor(xHost, {8}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_INT32, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeIamaxPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasIsamax(handle, 8, x, 2, result), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasIamaxStrideGuard, IcamaxRejectStridedIncx)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost = {{1.0f, 1.0f}, {2.0f, -2.0f}, {3.0f, 3.0f}, {-4.0f, 0.5f},
                                              {0.5f, 0.5f}, {1.0f, -1.0f}, {2.0f, 1.0f}, {0.0f, 0.25f}};
    std::vector<int32_t> resultHost(1, 0);
    ASSERT_EQ(CreateAclTensor(xHost, {8}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_INT32, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeIamaxPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdBlasIcamax(handle, 8, x, 2, result), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasIamaxFunc, IsamaxArgmaxBasic)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<float> xHost = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f, 4.5f, -1.5f, 0.25f};
    std::vector<int32_t> resultHost(1, 0);
    ASSERT_EQ(CreateAclTensor(xHost, {8}, &xAddr, aclDataType::ACL_FLOAT, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_INT32, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeIamaxPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    ASSERT_EQ(asdBlasGetWorkspaceSize(handle, workSize), AsdSip::ErrorType::ACL_SUCCESS);
    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ASSERT_EQ(aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST),
                  ::ACL_SUCCESS);
        ASSERT_EQ(asdBlasSetWorkspace(handle, workspaceAddr), AsdSip::ErrorType::ACL_SUCCESS);
    }
    ASSERT_EQ(asdBlasSetStream(handle, static_cast<aclrtStream>(stream)), AsdSip::ErrorType::ACL_SUCCESS);

    EXPECT_EQ(asdBlasIsamax(handle, 8, x, 1, result), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasSynchronize(handle), AsdSip::ErrorType::ACL_SUCCESS);

    std::vector<int32_t> resultOut(1, 0);
    ASSERT_EQ(aclrtMemcpy(resultOut.data(), sizeof(int32_t), resultAddr, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
              ::ACL_SUCCESS);
    EXPECT_EQ(resultOut[0], 6);

    asdBlasDestroy(handle);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestBlasIamaxFunc, IcamaxArgmaxBasic)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* x = nullptr;
    aclTensor* result = nullptr;
    void* xAddr = nullptr;
    void* resultAddr = nullptr;
    std::vector<std::complex<float>> xHost = {{1.0f, 1.0f}, {2.0f, -2.0f}, {3.0f, 3.0f}, {-4.0f, 0.5f},
                                              {0.5f, 0.5f}, {1.0f, -1.0f}, {2.0f, 1.0f}, {0.0f, 0.25f}};
    std::vector<int32_t> resultHost(1, 0);
    ASSERT_EQ(CreateAclTensor(xHost, {8}, &xAddr, aclDataType::ACL_COMPLEX64, &x), 0);
    ASSERT_EQ(CreateAclTensor(resultHost, {1}, &resultAddr, aclDataType::ACL_INT32, &result), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeIamaxPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    ASSERT_EQ(asdBlasGetWorkspaceSize(handle, workSize), AsdSip::ErrorType::ACL_SUCCESS);
    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ASSERT_EQ(aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST),
                  ::ACL_SUCCESS);
        ASSERT_EQ(asdBlasSetWorkspace(handle, workspaceAddr), AsdSip::ErrorType::ACL_SUCCESS);
    }
    ASSERT_EQ(asdBlasSetStream(handle, static_cast<aclrtStream>(stream)), AsdSip::ErrorType::ACL_SUCCESS);

    EXPECT_EQ(asdBlasIcamax(handle, 8, x, 1, result), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasSynchronize(handle), AsdSip::ErrorType::ACL_SUCCESS);

    std::vector<int32_t> resultOut(1, 0);
    ASSERT_EQ(aclrtMemcpy(resultOut.data(), sizeof(int32_t), resultAddr, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
              ::ACL_SUCCESS);
    EXPECT_EQ(resultOut[0], 3);

    asdBlasDestroy(handle);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyAclTensorWithAddr(x, xAddr);
    DestroyAclTensorWithAddr(result, resultAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}
