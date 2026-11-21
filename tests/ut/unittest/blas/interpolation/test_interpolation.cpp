/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// rsInterpolationBySinc 守卫层用例（issue #464，恢复 blas/interpolation UT 注册）：
// 正向：16/32 固定布局 + interpLength 为 16 的倍数 + 零输入（内核对 x 全程线性，
// x=0 时输出必为全 0；输出缓冲以哨兵值预填后断言全 0，可同时证明内核端到端执行
// 且写出正确值）；
// 负向：interpNum/quantNum 非 16/32 固定布局（issue #458）、interpLength 非 16 倍数
// 与 0（issue #457）、0 维张量。

#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "domain/rs_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;

namespace {

constexpr int TAB_INTERP_NUM = 16;
constexpr int TAB_QUANT_NUM = 32;
// sincTab 元素数：4 * ((quantNum + 1) * 2) * (interpNum * 2 + 8) = 10560
constexpr int64_t SINC_TAB_NUM = 4 * ((TAB_QUANT_NUM + 1) * 2) * (TAB_INTERP_NUM * 2 + 8);

struct InterpTensorBundle {
    aclTensor* input = nullptr;
    aclTensor* sincTab = nullptr;
    aclTensor* posFloor = nullptr;
    aclTensor* posToTabIndex = nullptr;
    aclTensor* output = nullptr;
    void* inputAddr = nullptr;
    void* sincTabAddr = nullptr;
    void* posFloorAddr = nullptr;
    void* posToTabIndexAddr = nullptr;
    void* outputAddr = nullptr;
};

int CreateInterpTensors(InterpTensorBundle& bundle, int64_t batch, int64_t signalLength, int interpLength,
                        const std::complex<float>& inputVal, const std::complex<float>& outputVal)
{
    std::vector<std::complex<float>> inputHost(batch * signalLength, inputVal);
    std::vector<float> sincTabHost(SINC_TAB_NUM, 1.0f);
    std::vector<int32_t> posFloorHost(batch * interpLength, 0);
    std::vector<int16_t> posToTabIndexHost(batch * interpLength, 0);
    std::vector<std::complex<float>> outputHost(batch * interpLength, outputVal);
    int ret = CreateAclTensor(inputHost, {batch, signalLength}, &bundle.inputAddr, aclDataType::ACL_COMPLEX64,
                              &bundle.input);
    if (ret != 0) {
        return ret;
    }
    ret = CreateAclTensor(sincTabHost, {1, SINC_TAB_NUM}, &bundle.sincTabAddr, aclDataType::ACL_FLOAT, &bundle.sincTab);
    if (ret != 0) {
        return ret;
    }
    ret = CreateAclTensor(posFloorHost, {batch, interpLength}, &bundle.posFloorAddr, aclDataType::ACL_INT32,
                          &bundle.posFloor);
    if (ret != 0) {
        return ret;
    }
    ret = CreateAclTensor(posToTabIndexHost, {batch, interpLength}, &bundle.posToTabIndexAddr, aclDataType::ACL_INT16,
                          &bundle.posToTabIndex);
    if (ret != 0) {
        return ret;
    }
    return CreateAclTensor(outputHost, {batch, interpLength}, &bundle.outputAddr, aclDataType::ACL_COMPLEX64,
                           &bundle.output);
}

void DestroyInterpTensors(const InterpTensorBundle& bundle)
{
    if (bundle.input != nullptr) {
        aclDestroyTensor(bundle.input);
    }
    if (bundle.sincTab != nullptr) {
        aclDestroyTensor(bundle.sincTab);
    }
    if (bundle.posFloor != nullptr) {
        aclDestroyTensor(bundle.posFloor);
    }
    if (bundle.posToTabIndex != nullptr) {
        aclDestroyTensor(bundle.posToTabIndex);
    }
    if (bundle.output != nullptr) {
        aclDestroyTensor(bundle.output);
    }
    if (bundle.inputAddr != nullptr) {
        aclrtFree(bundle.inputAddr);
    }
    if (bundle.sincTabAddr != nullptr) {
        aclrtFree(bundle.sincTabAddr);
    }
    if (bundle.posFloorAddr != nullptr) {
        aclrtFree(bundle.posFloorAddr);
    }
    if (bundle.posToTabIndexAddr != nullptr) {
        aclrtFree(bundle.posToTabIndexAddr);
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

// 正向：16/32 固定布局 + interpLength=16 + 零输入，输出应为全 0
TEST(TestRsInterpolationBySinc, ZeroInputZeroOutput)
{
    MkiRtStream stream = OpTestInit(0);
    const int64_t batch = 1;
    const int64_t signalLength = 16;
    const int interpLength = 16;
    InterpTensorBundle bundle;
    ASSERT_EQ(CreateInterpTensors(bundle, batch, signalLength, interpLength, {0.0f, 0.0f}, {7.0f, 7.0f}), 0);

    size_t workSize = 0;
    ASSERT_EQ(rsInterpolationBySincGetWorkspaceSize(workSize), AsdSip::ErrorType::ACL_SUCCESS);
    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ASSERT_EQ(aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST),
                  ::ACL_SUCCESS);
    }

    EXPECT_EQ(
        rsInterpolationBySinc(bundle.input, bundle.sincTab, bundle.posFloor, bundle.posToTabIndex, bundle.output,
                              TAB_INTERP_NUM, TAB_QUANT_NUM, interpLength, static_cast<void*>(stream), workspaceAddr),
        AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)), ::ACL_SUCCESS);

    std::vector<std::complex<float>> outputAfter(batch * interpLength, {7.0f, 7.0f});
    ASSERT_EQ(aclrtMemcpy(outputAfter.data(), static_cast<int64_t>(outputAfter.size() * sizeof(std::complex<float>)),
                          bundle.outputAddr, static_cast<int64_t>(outputAfter.size() * sizeof(std::complex<float>)),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ::ACL_SUCCESS);
    for (const auto& val : outputAfter) {
        ASSERT_EQ(val, std::complex<float>(0.0f, 0.0f));
    }

    DestroyInterpTensors(bundle);
    DestroyWorkspace(workspaceAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// tab 布局约束（issue #458）：内核 sincTab 打包布局按 16/32 固定排布，其余组合显式拒绝
TEST(TestRsInterpolationBySincGuard, RejectTabLayoutNotFixed)
{
    MkiRtStream stream = OpTestInit(0);
    InterpTensorBundle bundle;
    ASSERT_EQ(CreateInterpTensors(bundle, 1, 16, 16, {1.0f, 0.0f}, {0.0f, 0.0f}), 0);

    EXPECT_EQ(rsInterpolationBySinc(bundle.input, bundle.sincTab, bundle.posFloor, bundle.posToTabIndex, bundle.output,
                                    8, TAB_QUANT_NUM, 16, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(rsInterpolationBySinc(bundle.input, bundle.sincTab, bundle.posFloor, bundle.posToTabIndex, bundle.output,
                                    17, TAB_QUANT_NUM, 16, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(rsInterpolationBySinc(bundle.input, bundle.sincTab, bundle.posFloor, bundle.posToTabIndex, bundle.output,
                                    TAB_INTERP_NUM, 3, 16, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(rsInterpolationBySinc(bundle.input, bundle.sincTab, bundle.posFloor, bundle.posToTabIndex, bundle.output,
                                    TAB_INTERP_NUM, 8, 16, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    DestroyInterpTensors(bundle);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// interpLength 约束（issue #457）：非 16 倍数与 0 显式拒绝，防尾块静默不计算
TEST(TestRsInterpolationBySincGuard, RejectInterpLengthNotMultipleOf16)
{
    MkiRtStream stream = OpTestInit(0);
    InterpTensorBundle bundle;
    ASSERT_EQ(CreateInterpTensors(bundle, 1, 16, 16, {1.0f, 0.0f}, {0.0f, 0.0f}), 0);

    EXPECT_EQ(rsInterpolationBySinc(bundle.input, bundle.sincTab, bundle.posFloor, bundle.posToTabIndex, bundle.output,
                                    TAB_INTERP_NUM, TAB_QUANT_NUM, 20, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(rsInterpolationBySinc(bundle.input, bundle.sincTab, bundle.posFloor, bundle.posToTabIndex, bundle.output,
                                    TAB_INTERP_NUM, TAB_QUANT_NUM, 0, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    DestroyInterpTensors(bundle);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// 0 维张量守卫（存量口径覆盖）：0 维输入返回 OP_INPUT_NOT_MATCH，不崩溃
TEST(TestRsInterpolationBySincGuard, RejectZeroDimInput)
{
    MkiRtStream stream = OpTestInit(0);
    InterpTensorBundle bundle;
    ASSERT_EQ(CreateInterpTensors(bundle, 1, 16, 16, {1.0f, 0.0f}, {0.0f, 0.0f}), 0);
    void* dim0Addr = nullptr;
    auto ret = aclrtMalloc(&dim0Addr, 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    aclDestroyTensor(bundle.input);
    bundle.input = aclCreateTensor(nullptr, 0, aclDataType::ACL_COMPLEX64, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                   nullptr, 0, dim0Addr);
    ASSERT_NE(bundle.input, nullptr);

    EXPECT_EQ(rsInterpolationBySinc(bundle.input, bundle.sincTab, bundle.posFloor, bundle.posToTabIndex, bundle.output,
                                    TAB_INTERP_NUM, TAB_QUANT_NUM, 16, static_cast<void*>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    DestroyInterpTensors(bundle);
    DestroyWorkspace(dim0Addr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}
