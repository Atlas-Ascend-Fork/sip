/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <complex>
#include <cstdint>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "fft_api.h"
#include "acl/acl.h"
using namespace AsdSip;
using namespace Mki;

// fftshift/ifftshift 乘法溢出防护负向用例（issue #354）。
// 说明：fft/fftshift 与 fft/ifftshift 目录下的功能用例（含 2.4GB 级 MRI 大用例）未注册进
// ops_unittest，本文件仅注册轻量负向校验，不改变存量目录的注册范围。

namespace {

void RunShiftOverflowCase(bool isIfftShift)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);

    std::vector<std::complex<float>> inData(16, std::complex<float>(1.0f, 0.0f));
    std::vector<std::complex<float>> outData(16, std::complex<float>(0.0f, 0.0f));
    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(inData, {16}, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput), 0);
    ASSERT_EQ(CreateAclTensor(outData, {16}, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutput), 0);

    constexpr int64_t kBig = 1LL << 32; // 2^32 * 2^32 回绕为 0
    if (isIfftShift) {
        EXPECT_EQ(asdIfftShift(kBig, kBig, 1, 0, 1.0, aclInput, aclOutput, aclStream, nullptr),
                  AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        EXPECT_EQ(asdIfftShift(1, 1, 1LL << 32, 1LL << 33, 1.0, aclInput, aclOutput, aclStream, nullptr),
                  AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    } else {
        EXPECT_EQ(asdFftShift(kBig, kBig, 1, 0, aclInput, aclOutput, aclStream, nullptr),
                  AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        EXPECT_EQ(asdFftShift(1, 1, 1LL << 32, 1LL << 33, aclInput, aclOutput, aclStream, nullptr),
                  AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    }

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

} // namespace

// row*col 回绕为 0 时形状校验恒过、内核按真实尺寸越界寻址；修复后主机侧应显式拒绝
TEST(TestFftShiftNegative, TestShiftRowColOverflowRejected) { RunShiftOverflowCase(false); }

// (batch-1)*dist + row*col 回绕为负时 requiredInNum 校验恒过；修复后应显式拒绝
TEST(TestFftShiftNegative, TestShiftRequiredInOverflowRejected) { RunShiftOverflowCase(false); }

// ---- 负向校验：张量侧维度连乘回绕（issue #369）----
// GetTensorNum 对 storage shape 各维连乘无溢出防护时，乘积回绕为"大正值"会使
// inNumOpt/outNumOpt 校验被绕过；修复后 GetTensorNum 返回 nullopt，入口拒绝。
// 张量仅作描述符（aclCreateTensor 不按 shape 分配真实内存），绑定小缓冲即可，
// 用例在形状校验阶段返回，不触发内核。

// 构造回绕 storage shape 的描述符张量（复数 64bit，元数据指向 64 字节真实缓冲）
aclTensor* CreateWrappedShapeTensor(const std::vector<int64_t>& dims, void* deviceBuf)
{
    std::vector<int64_t> strides = {dims[1], 1};
    return aclCreateTensor(dims.data(), dims.size(), ACL_COMPLEX64, strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                           dims.size(), deviceBuf);
}

// inTensor 维度连乘回绕为大正值（(2^40+1)*2^24 mod 2^64 = 2^24），requiredInNum=10^6
// 被回绕值满足，修复前校验恒过并按 10^6 元素寻址真实 64 字节缓冲越界读
TEST(TestFftShiftNegative, TestShiftTensorDimOverflowRejected)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);

    void* tinyBuf = nullptr;
    ASSERT_EQ(aclrtMalloc(&tinyBuf, 64, ACL_MEM_MALLOC_HUGE_FIRST), ::ACL_SUCCESS);

    // 回绕 shape 的输入描述符 + 正常小输出
    aclTensor* aclIn = CreateWrappedShapeTensor({(1LL << 40) + 1, 1LL << 24}, tinyBuf);
    ASSERT_NE(aclIn, nullptr);
    std::vector<std::complex<float>> outData(16, std::complex<float>(0.0f, 0.0f));
    aclTensor* aclOut = nullptr;
    void* outDeviceAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(outData, {16}, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOut), 0);

    EXPECT_EQ(asdFftShift(1000, 1000, 1, 0, aclIn, aclOut, aclStream, nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    aclDestroyTensor(aclIn);
    aclDestroyTensor(aclOut);
    aclrtFree(tinyBuf);
    aclrtFree(outDeviceAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// outTensor 维度连乘回绕（修复前 outNumOpt 回绕值满足 requiredOutNum，写越界）
TEST(TestFftShiftNegative, TestShiftTensorDimOverflowOutRejected)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);

    void* tinyBuf = nullptr;
    ASSERT_EQ(aclrtMalloc(&tinyBuf, 64, ACL_MEM_MALLOC_HUGE_FIRST), ::ACL_SUCCESS);

    // 正常小输入（形状元素数与 row/col/batch 匹配）
    std::vector<std::complex<float>> inData(16, std::complex<float>(1.0f, 0.0f));
    aclTensor* aclIn = nullptr;
    void* inDeviceAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(inData, {16}, &inDeviceAddr, aclDataType::ACL_COMPLEX64, &aclIn), 0);
    // 回绕 shape 的输出描述符
    aclTensor* aclOut = CreateWrappedShapeTensor({(1LL << 40) + 1, 1LL << 24}, tinyBuf);
    ASSERT_NE(aclOut, nullptr);

    EXPECT_EQ(asdFftShift(2, 2, 4, 0, aclIn, aclOut, aclStream, nullptr), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    aclDestroyTensor(aclIn);
    aclDestroyTensor(aclOut);
    aclrtFree(tinyBuf);
    aclrtFree(inDeviceAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// ifftshift（含 scale 路径）同款防护
TEST(TestIfftShiftNegative, TestIfftShiftRowColOverflowRejected) { RunShiftOverflowCase(true); }
TEST(TestIfftShiftNegative, TestIfftShiftRequiredInOverflowRejected) { RunShiftOverflowCase(true); }
