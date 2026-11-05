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

// ifftshift（含 scale 路径）同款防护
TEST(TestIfftShiftNegative, TestIfftShiftRowColOverflowRejected) { RunShiftOverflowCase(true); }
TEST(TestIfftShiftNegative, TestIfftShiftRequiredInOverflowRejected) { RunShiftOverflowCase(true); }
