/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// asdInterpWithCoeff（InterpByCoeff）守卫层用例（issue #464，恢复 blas/interpolation UT 注册）：
// 正向：零输入 x（内核为 batched matmul，对 x 全程线性，x=0 时输出必为全 0；输出缓冲
// 以哨兵值预填后断言全 0，可同时证明内核端到端执行且写出正确值）；
// 负向：x/coefficient/output 三张量 dtype 不一致（issue #459）、0 维张量。

#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "interp_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;

namespace {

struct InterpCoeffTensorBundle {
    aclTensor* x = nullptr;
    aclTensor* coeff = nullptr;
    aclTensor* output = nullptr;
    void* xAddr = nullptr;
    void* coeffAddr = nullptr;
    void* outputAddr = nullptr;
};

int CreateInterpCoeffTensors(InterpCoeffTensorBundle& bundle, aclDataType coeffDtype, aclDataType outputDtype,
                             const std::complex<float>& xVal, const std::complex<float>& outputVal)
{
    const int64_t batch = 1;
    const int64_t rsNum = 2;
    const int64_t subcarrier = 8;
    const int64_t interpLength = 2;
    std::vector<std::complex<float>> xHost(batch * rsNum * subcarrier, xVal);
    std::vector<std::complex<float>> coeffHost(batch * interpLength * rsNum, {1.0f, 0.5f});
    std::vector<std::complex<float>> outputHost(batch * interpLength * subcarrier, outputVal);
    int ret = CreateAclTensor(xHost, {batch, rsNum, subcarrier}, &bundle.xAddr, aclDataType::ACL_COMPLEX64, &bundle.x);
    if (ret != 0) {
        return ret;
    }
    ret = CreateAclTensor(coeffHost, {batch, interpLength, rsNum}, &bundle.coeffAddr, coeffDtype, &bundle.coeff);
    if (ret != 0) {
        return ret;
    }
    return CreateAclTensor(outputHost, {batch, interpLength, subcarrier}, &bundle.outputAddr, outputDtype,
                           &bundle.output);
}

void DestroyInterpCoeffTensors(const InterpCoeffTensorBundle& bundle)
{
    if (bundle.x != nullptr) {
        aclDestroyTensor(bundle.x);
    }
    if (bundle.coeff != nullptr) {
        aclDestroyTensor(bundle.coeff);
    }
    if (bundle.output != nullptr) {
        aclDestroyTensor(bundle.output);
    }
    if (bundle.xAddr != nullptr) {
        aclrtFree(bundle.xAddr);
    }
    if (bundle.coeffAddr != nullptr) {
        aclrtFree(bundle.coeffAddr);
    }
    if (bundle.outputAddr != nullptr) {
        aclrtFree(bundle.outputAddr);
    }
}

void DestroyWorkspace(void* workspaceAddr)
{
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
}

} // namespace

// 正向：零输入 x，输出应为全 0
TEST(TestAsdInterpWithCoeff, ZeroInputZeroOutput)
{
    MkiRtStream stream = OpTestInit(0);
    const int64_t batch = 1;
    const int64_t interpLength = 2;
    const int64_t subcarrier = 8;
    InterpCoeffTensorBundle bundle;
    ASSERT_EQ(CreateInterpCoeffTensors(bundle, aclDataType::ACL_COMPLEX64, aclDataType::ACL_COMPLEX64, {0.0f, 0.0f},
                                       {7.0f, 7.0f}),
              0);

    size_t workSize = 0;
    ASSERT_EQ(asdInterpWithCoeffGetWorkspaceSize(workSize), AsdSip::ErrorType::ACL_SUCCESS);
    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ASSERT_EQ(aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST),
                  ::ACL_SUCCESS);
    }

    EXPECT_EQ(asdInterpWithCoeff(bundle.x, bundle.coeff, bundle.output, static_cast<void*>(stream), workspaceAddr),
              AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)), ::ACL_SUCCESS);

    std::vector<std::complex<float>> outputAfter(batch * interpLength * subcarrier, {7.0f, 7.0f});
    ASSERT_EQ(aclrtMemcpy(outputAfter.data(), static_cast<int64_t>(outputAfter.size() * sizeof(std::complex<float>)),
                          bundle.outputAddr, static_cast<int64_t>(outputAfter.size() * sizeof(std::complex<float>)),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ::ACL_SUCCESS);
    for (const auto& val : outputAfter) {
        ASSERT_EQ(val, std::complex<float>(0.0f, 0.0f));
    }

    DestroyInterpCoeffTensors(bundle);
    DestroyWorkspace(workspaceAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// dtype 一致性校验（issue #459）：coefficient/output 与 x dtype 不一致显式拒绝，
// 防按错误元素宽度解读/写入缓冲静默算错
TEST(TestAsdInterpWithCoeffGuard, RejectDtypeMismatch)
{
    MkiRtStream stream = OpTestInit(0);
    InterpCoeffTensorBundle bundle;
    ASSERT_EQ(CreateInterpCoeffTensors(bundle, aclDataType::ACL_COMPLEX64, aclDataType::ACL_COMPLEX64, {1.0f, 0.0f},
                                       {0.0f, 0.0f}),
              0);

    // coefficient 为 COMPLEX32：与 x（COMPLEX64）不一致
    InterpCoeffTensorBundle coeffHalfBundle;
    ASSERT_EQ(CreateInterpCoeffTensors(coeffHalfBundle, aclDataType::ACL_COMPLEX32, aclDataType::ACL_COMPLEX64,
                                       {1.0f, 0.0f}, {0.0f, 0.0f}),
              0);
    EXPECT_EQ(asdInterpWithCoeff(bundle.x, coeffHalfBundle.coeff, bundle.output, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    DestroyInterpCoeffTensors(coeffHalfBundle);

    // output 为 COMPLEX32：与 x（COMPLEX64）不一致
    InterpCoeffTensorBundle outputHalfBundle;
    ASSERT_EQ(CreateInterpCoeffTensors(outputHalfBundle, aclDataType::ACL_COMPLEX64, aclDataType::ACL_COMPLEX32,
                                       {1.0f, 0.0f}, {0.0f, 0.0f}),
              0);
    EXPECT_EQ(asdInterpWithCoeff(bundle.x, bundle.coeff, outputHalfBundle.output, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    DestroyInterpCoeffTensors(outputHalfBundle);

    DestroyInterpCoeffTensors(bundle);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// 0 维张量守卫（存量口径覆盖）：0 维输入返回 OP_INPUT_NOT_MATCH，不崩溃
TEST(TestAsdInterpWithCoeffGuard, RejectZeroDimInput)
{
    MkiRtStream stream = OpTestInit(0);
    InterpCoeffTensorBundle bundle;
    ASSERT_EQ(CreateInterpCoeffTensors(bundle, aclDataType::ACL_COMPLEX64, aclDataType::ACL_COMPLEX64, {1.0f, 0.0f},
                                       {0.0f, 0.0f}),
              0);
    void* dim0Addr = nullptr;
    auto ret = aclrtMalloc(&dim0Addr, 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    aclDestroyTensor(bundle.x);
    bundle.x = aclCreateTensor(nullptr, 0, aclDataType::ACL_COMPLEX64, nullptr, 0, aclFormat::ACL_FORMAT_ND, nullptr, 0,
                               dim0Addr);
    ASSERT_NE(bundle.x, nullptr);

    EXPECT_EQ(asdInterpWithCoeff(bundle.x, bundle.coeff, bundle.output, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    DestroyInterpCoeffTensors(bundle);
    DestroyWorkspace(dim0Addr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}
