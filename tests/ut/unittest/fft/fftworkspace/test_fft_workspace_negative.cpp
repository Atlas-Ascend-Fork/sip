/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// FFT workspace 需求量守卫负向用例（issue #486）：
// 跳过 asdFftGetWorkspaceSize 直接 SetWorkspace+Exec：plan.workspaceSize 为 0，
// Workspace 以 capacity_==0"容量未知"语义构造，#195 越界防护被整体旁路，内核对
// 用户小缓冲静默越界写；修复后 exec 返回 ACL_ERROR_INVALID_PARAM。
// 说明：asdFftSetWorkspace 仅接收指针、无尺寸入参，"绑定小于需求量缓冲"场景在
// 现有 C API 下无法在主机侧识别（#195 的 capacity 取 GetWorkspaceSize 记录的需求量，
// 内核分配量恒不超需求量故不触发越界抛出），该子项需 API 演进（SetWorkspace 增加
// 尺寸入参）才能落地，本 PR 不改 API，详见 PR 说明。

#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "fft_api.h"
#include "acl/acl.h"

using namespace AsdSip;
using namespace Mki;

namespace {

constexpr int64_t kSizeX = 16;
constexpr int64_t kSizeY = 16;

void CreateC2C2DTensors(aclTensor** aclInput, aclTensor** aclOutput, void** inAddr, void** outAddr)
{
    std::vector<std::complex<float>> inData(kSizeX * kSizeY, std::complex<float>(1.0f, 0.0f));
    std::vector<std::complex<float>> outData(kSizeX * kSizeY, std::complex<float>(0.0f, 0.0f));
    ASSERT_EQ(CreateAclTensor(inData, {kSizeX, kSizeY}, inAddr, aclDataType::ACL_COMPLEX64, aclInput), 0);
    ASSERT_EQ(CreateAclTensor(outData, {kSizeX, kSizeY}, outAddr, aclDataType::ACL_COMPLEX64, aclOutput), 0);
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

// 跳过 GetWorkspaceSize 直接 SetWorkspace+Exec：需求量未记录（workspaceSize==0），
// 修复前以 capacity 未知语义放行，内核对小缓冲静默越界写
TEST(TestFftWorkspaceGuard, CApiExecRejectUnqueriedWorkspaceSize)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);

    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inAddr = nullptr;
    void* outAddr = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateC2C2DTensors(&aclInput, &aclOutput, &inAddr, &outAddr));

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(
        asdFftMakePlan2D(handle, kSizeX, kSizeY, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
        AsdSip::ErrorType::ACL_SUCCESS);

    // 故意跳过 asdFftGetWorkspaceSize，直接绑定一块小缓冲
    void* smallWorkspace = nullptr;
    ASSERT_EQ(aclrtMalloc(&smallWorkspace, 64, ACL_MEM_MALLOC_HUGE_FIRST), ::ACL_SUCCESS);
    ASSERT_EQ(asdFftSetWorkspace(handle, smallWorkspace), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftSetStream(handle, aclStream), AsdSip::ErrorType::ACL_SUCCESS);

    EXPECT_EQ(asdFftExecC2C(handle, aclInput, aclOutput), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdFftDestroy(handle);
    aclrtFree(smallWorkspace);
    DestroyAclTensorWithAddr(aclInput, inAddr);
    DestroyAclTensorWithAddr(aclOutput, outAddr);
    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}
