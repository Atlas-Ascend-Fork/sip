/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// asdConvolve kernel 维度契约负向用例（issue #387）：文档约定滤波系数 shape 为 [k]（一维），
// 原实现 1/2 维均放行且无条件取 storageDims[0]，2 维 kernel 被静默按第 0 维处理，结果整体
// 错误且无报错；修复后非一维 kernel 应返回 ACL_ERROR_OP_INPUT_NOT_MATCH。

#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "filter_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;

namespace {

constexpr int64_t SIGNAL_LEN = 16;

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

// 2 维 kernel（如 {batchCount, k}）应被显式拒绝，不再静默取 dims[0]
TEST(TestConvolveKernelDimGuard, RejectTwoDimKernel)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* signal = nullptr;
    aclTensor* kernel = nullptr;
    aclTensor* output = nullptr;
    void* signalAddr = nullptr;
    void* kernelAddr = nullptr;
    void* outputAddr = nullptr;
    std::vector<std::complex<float>> signalHost(SIGNAL_LEN, {1.0f, 0.0f});
    std::vector<float> kernelHost(8, 1.0f);
    std::vector<std::complex<float>> outputHost(SIGNAL_LEN, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(signalHost, {1, SIGNAL_LEN}, &signalAddr, aclDataType::ACL_COMPLEX64, &signal), 0);
    ASSERT_EQ(CreateAclTensor(kernelHost, {2, 4}, &kernelAddr, aclDataType::ACL_FLOAT, &kernel), 0);
    ASSERT_EQ(CreateAclTensor(outputHost, {1, SIGNAL_LEN}, &outputAddr, aclDataType::ACL_COMPLEX64, &output), 0);

    EXPECT_EQ(
        asdConvolve(signal, kernel, output, asdConvolveMode_t::ASD_CONVOLVE_SAME, static_cast<void*>(stream), nullptr),
        AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    DestroyAclTensorWithAddr(signal, signalAddr);
    DestroyAclTensorWithAddr(kernel, kernelAddr);
    DestroyAclTensorWithAddr(output, outputAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}
