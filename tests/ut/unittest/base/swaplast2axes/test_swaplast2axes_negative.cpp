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
#include "base_api.h"
#include "acl/acl.h"
using namespace AsdSip;
using namespace Mki;

// swapLast2Axes 张量维度连乘回绕负向用例（issue #369）：GetTensorNum 对 storage
// shape 各维连乘无溢出防护时，乘积回绕为"大正值"会使 outNumOpt >= inNumOpt 校验
// 被绕过；修复后 GetTensorNum 对回绕 shape 返回 nullopt，入口在形状校验阶段拒绝，
// 不触发内核。张量仅作描述符（aclCreateTensor 不按 shape 分配真实内存）。

namespace {

// 构造回绕 storage shape 的描述符张量（复数 64bit，元数据指向 64 字节真实缓冲）
aclTensor* CreateWrappedShapeTensor(const std::vector<int64_t>& dims, void* deviceBuf)
{
    std::vector<int64_t> strides = {dims[1], 1};
    return aclCreateTensor(dims.data(), dims.size(), ACL_COMPLEX64, strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                           dims.size(), deviceBuf);
}

// in/out 张量维度连乘均回绕（(2^40+1)*2^24 mod 2^64 = 2^24）：修复前回绕值使
// outNumOpt >= inNumOpt 恒成立，内核按未回绕尺寸寻址真实 64 字节缓冲越界读写
TEST(TestSwapLast2AxesNegative, TestTensorDimOverflowRejected)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);

    void* tinyBuf = nullptr;
    ASSERT_EQ(aclrtMalloc(&tinyBuf, 64, ACL_MEM_MALLOC_HUGE_FIRST), ::ACL_SUCCESS);

    aclTensor* aclIn = CreateWrappedShapeTensor({(1LL << 40) + 1, 1LL << 24}, tinyBuf);
    ASSERT_NE(aclIn, nullptr);
    aclTensor* aclOut = CreateWrappedShapeTensor({(1LL << 40) + 1, 1LL << 24}, tinyBuf);
    ASSERT_NE(aclOut, nullptr);

    EXPECT_EQ(swapLast2Axes(aclIn, aclOut, aclStream, nullptr), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    aclDestroyTensor(aclIn);
    aclDestroyTensor(aclOut);
    aclrtFree(tinyBuf);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}
} // namespace
