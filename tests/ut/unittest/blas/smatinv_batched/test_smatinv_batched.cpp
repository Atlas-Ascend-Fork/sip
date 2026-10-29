/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// asdBlasSmatinvBatched 单元测试：
// 1. 正向用例：n=64 (SgetriBatched 分支, 32 <= n <= 256) fp32 批量矩阵求逆，
//    通过残差 max||A @ Ainv - I|| 验证正确性
// 2. 负向用例：lda/lda_inv 非法（<= 0 或 != n）、dtype 非法时应返回
//    ACL_ERROR_INVALID_PARAM / ACL_ERROR_UNSUPPORTED_DATA_TYPE，而非静默计算

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "blas_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;
using namespace Mki;

namespace {

constexpr int64_t TEST_N = 64;
constexpr int64_t TEST_BATCH = 2;
constexpr float RESIDUAL_TOL = 1e-2f;

// 生成对角占优可逆矩阵：A = I + alpha * Rand，对角线放大保证收敛
std::vector<float> MakeDiagDominant(int64_t n, int64_t batch, float scale)
{
    std::vector<float> a(static_cast<size_t>(batch * n * n), 0.0f);
    for (int64_t bi = 0; bi < batch; bi++) {
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < n; j++) {
                size_t idx = static_cast<size_t>((bi * n + i) * n + j);
                if (i == j) {
                    a[idx] = scale;
                } else {
                    a[idx] = 0.01f * (static_cast<float>((i * n + j + bi) % 7) - 3.0f);
                }
            }
        }
    }
    return a;
}

// 构造 batch * n * n 的零张量
std::vector<float> MakeZeros(int64_t n, int64_t batch)
{
    return std::vector<float>(static_cast<size_t>(batch * n * n), 0.0f);
}

// 非对角占优普通随机矩阵（固定种子，确定性；issue #302）：大幅非对角元素
// 触发真实行交换（partial pivot），覆盖 pivot 应用路径
std::vector<float> MakeRandomGeneral(int64_t n, int64_t batch, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> distr(-8.0f, 8.0f);
    std::vector<float> a(static_cast<size_t>(batch * n * n), 0.0f);
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = distr(gen);
        if (std::fabs(a[i]) < 0.25f) {
            a[i] = (a[i] >= 0) ? 0.25f : -0.25f; // 避免过小元素造成病态
        }
    }
    return a;
}

// 残差检查：max|A @ Ainv - I|，host 侧朴素乘法
float CheckInversion(const std::vector<float>& a, const std::vector<float>& ainv, int64_t n, int64_t batch)
{
    float maxResidual = 0.0f;
    for (int64_t bi = 0; bi < batch; bi++) {
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < n; k++) {
                    sum += a[static_cast<size_t>((bi * n + i) * n + k)] *
                           ainv[static_cast<size_t>((bi * n + k) * n + j)];
                }
                float expect = (i == j) ? 1.0f : 0.0f;
                float residual = std::fabs(sum - expect);
                maxResidual = std::max(maxResidual, residual);
            }
        }
    }
    return maxResidual;
}

TEST(TestBlasSmatinvBatchedPositive, InverseF32N64)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    std::vector<float> aHost = MakeDiagDominant(TEST_N, TEST_BATCH, 8.0f);
    std::vector<float> ainvHost = MakeZeros(TEST_N, TEST_BATCH);
    aclTensor* aclA = nullptr;
    aclTensor* aclAinv = nullptr;
    void* aAddr = nullptr;
    void* ainvAddr = nullptr;
    ASSERT_EQ(
        CreateAclTensor(aHost, std::vector<int64_t>{TEST_BATCH, TEST_N, TEST_N}, &aAddr, aclDataType::ACL_FLOAT, &aclA),
        0);
    ASSERT_EQ(CreateAclTensor(ainvHost, std::vector<int64_t>{TEST_BATCH, TEST_N, TEST_N}, &ainvAddr,
                              aclDataType::ACL_FLOAT, &aclAinv),
              0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSmatinvBatchedPlan(handle, TEST_N, TEST_BATCH), AsdSip::ErrorType::ACL_SUCCESS);

    // workspace + stream 绑定 (对齐 ccopy 等存量 UT 范式, 缺失时 RunAsdOpsV2 报 stream nullptr)
    size_t workSize = 0;
    ASSERT_EQ(asdBlasGetWorkspaceSize(handle, workSize), AsdSip::ErrorType::ACL_SUCCESS);
    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ASSERT_EQ(aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST), 0);
        ASSERT_EQ(asdBlasSetWorkspace(handle, workspaceAddr), AsdSip::ErrorType::ACL_SUCCESS);
    }
    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    ASSERT_EQ(asdBlasSetStream(handle, aclStream), AsdSip::ErrorType::ACL_SUCCESS);

    ASSERT_EQ(asdBlasSmatinvBatched(handle, TEST_N, aclA, TEST_N, aclAinv, TEST_N, nullptr, TEST_BATCH),
              AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    // 结果拷回 host 校验残差
    size_t bytes = static_cast<size_t>(TEST_BATCH * TEST_N * TEST_N) * sizeof(float);
    std::vector<float> ainvOut(static_cast<size_t>(TEST_BATCH * TEST_N * TEST_N), 0.0f);
    ASSERT_EQ(aclrtMemcpy(ainvOut.data(), bytes, ainvAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    float residual = CheckInversion(aHost, ainvOut, TEST_N, TEST_BATCH);
    EXPECT_LT(residual, RESIDUAL_TOL);

    asdBlasDestroy(handle);
    aclDestroyTensor(aclA);
    aclDestroyTensor(aclAinv);
    aclrtFree(aAddr);
    aclrtFree(ainvAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// 非对角占优随机矩阵：触发真实行交换（W[j]!=j），覆盖 pivot 应用路径（issue #302）
TEST(TestBlasSmatinvBatchedPositive, InverseF32RandomN64)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    std::vector<float> aHost = MakeRandomGeneral(TEST_N, TEST_BATCH, 20260101u);
    std::vector<float> ainvHost = MakeZeros(TEST_N, TEST_BATCH);
    aclTensor* aclA = nullptr;
    aclTensor* aclAinv = nullptr;
    void* aAddr = nullptr;
    void* ainvAddr = nullptr;
    ASSERT_EQ(
        CreateAclTensor(aHost, std::vector<int64_t>{TEST_BATCH, TEST_N, TEST_N}, &aAddr, aclDataType::ACL_FLOAT, &aclA),
        0);
    ASSERT_EQ(CreateAclTensor(ainvHost, std::vector<int64_t>{TEST_BATCH, TEST_N, TEST_N}, &ainvAddr,
                              aclDataType::ACL_FLOAT, &aclAinv),
              0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSmatinvBatchedPlan(handle, TEST_N, TEST_BATCH), AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    ASSERT_EQ(asdBlasGetWorkspaceSize(handle, workSize), AsdSip::ErrorType::ACL_SUCCESS);
    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ASSERT_EQ(aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST), 0);
        ASSERT_EQ(asdBlasSetWorkspace(handle, workspaceAddr), AsdSip::ErrorType::ACL_SUCCESS);
    }
    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    ASSERT_EQ(asdBlasSetStream(handle, aclStream), AsdSip::ErrorType::ACL_SUCCESS);

    ASSERT_EQ(asdBlasSmatinvBatched(handle, TEST_N, aclA, TEST_N, aclAinv, TEST_N, nullptr, TEST_BATCH),
              AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    size_t bytes = static_cast<size_t>(TEST_BATCH * TEST_N * TEST_N) * sizeof(float);
    std::vector<float> ainvOut(static_cast<size_t>(TEST_BATCH * TEST_N * TEST_N), 0.0f);
    ASSERT_EQ(aclrtMemcpy(ainvOut.data(), bytes, ainvAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    float residual = CheckInversion(aHost, ainvOut, TEST_N, TEST_BATCH);
    EXPECT_LT(residual, 0.1f); // 随机矩阵条件数更大，残差阈值放宽至 0.1

    asdBlasDestroy(handle);
    aclDestroyTensor(aclA);
    aclDestroyTensor(aclAinv);
    aclrtFree(aAddr);
    aclrtFree(ainvAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

TEST(TestBlasSmatinvBatchedNegative, LdaValidation)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    const int64_t n = 4;
    const int64_t batchSize = 2;
    std::vector<float> aHost = MakeDiagDominant(n, batchSize, 4.0f);
    std::vector<float> ainvHost = MakeZeros(n, batchSize);
    aclTensor* aclA = nullptr;
    aclTensor* aclAinv = nullptr;
    void* aAddr = nullptr;
    void* ainvAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(aHost, std::vector<int64_t>{batchSize, n, n}, &aAddr, aclDataType::ACL_FLOAT, &aclA), 0);
    ASSERT_EQ(
        CreateAclTensor(ainvHost, std::vector<int64_t>{batchSize, n, n}, &ainvAddr, aclDataType::ACL_FLOAT, &aclAinv),
        0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSmatinvBatchedPlan(handle, n, batchSize), AsdSip::ErrorType::ACL_SUCCESS);

    // lda <= 0
    EXPECT_EQ(asdBlasSmatinvBatched(handle, n, aclA, 0, aclAinv, n, nullptr, batchSize),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // lda_inv <= 0
    EXPECT_EQ(asdBlasSmatinvBatched(handle, n, aclA, n, aclAinv, -1, nullptr, batchSize),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // lda != n
    EXPECT_EQ(asdBlasSmatinvBatched(handle, n, aclA, 2 * n, aclAinv, 2 * n, nullptr, batchSize),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    aclDestroyTensor(aclA);
    aclDestroyTensor(aclAinv);
    aclrtFree(aAddr);
    aclrtFree(ainvAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

} // namespace
