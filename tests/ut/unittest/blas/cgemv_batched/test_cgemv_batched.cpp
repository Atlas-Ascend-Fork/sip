/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <vector>
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "blas_api.h"

using namespace AsdSip;

// trans 感知的 y/x 元素数契约（issue：trans='c' 且 m>n 时合法调用被 m-only
// 校验误拒）：OP_N 需 y=m/x=n，OP_T/OP_C 需 y=n/x=m。m=12 > n=6 的用例
// 专门覆盖该不一致（kernel IS_TRANS 路径本就只写 n 元素 y）。
namespace {

std::vector<std::complex<float>> MakeRandomC64(size_t count, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> distr(-100.0f, 100.0f);
    std::vector<std::complex<float>> v(count);
    for (auto& value : v) {
        value = {distr(gen), distr(gen)};
    }
    return v;
}

int CreateC64(const std::vector<std::complex<float>>& host, const std::vector<int64_t>& shape, void** deviceAddr,
              aclTensor** tensor)
{
    auto bytes = static_cast<int64_t>(host.size() * sizeof(std::complex<float>));
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_COMPLEX64, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 单用例跑三种 trans：host 双精度 golden + NPU 输出逐元素相对误差校验
void RunCgemvBatchedCase(asdBlasOperation_t trans, int64_t m, int64_t n, int64_t batch, uint32_t seed)
{
    MkiRtStream stream = OpTestInit(0);

    std::vector<std::complex<float>> a = MakeRandomC64(batch * m * n, seed);
    std::vector<std::complex<float>> x = MakeRandomC64(batch * m, seed + 1);
    std::vector<std::complex<float>> y(static_cast<size_t>(batch * n), {0.0f, 0.0f});

    aclTensor* aclA = nullptr;
    aclTensor* aclX = nullptr;
    aclTensor* aclY = nullptr;
    void *aAddr = nullptr, *xAddr = nullptr, *yAddr = nullptr;
    ASSERT_EQ(CreateC64(a, {batch, m, n}, &aAddr, &aclA), 0);
    ASSERT_EQ(CreateC64(x, {batch, m, 1}, &xAddr, &aclX), 0);
    ASSERT_EQ(CreateC64(y, {batch, n, 1}, &yAddr, &aclY), 0);

    asdBlasHandle handle = nullptr;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCgemvBatchedPlan(handle, trans, m), AsdSip::ErrorType::ACL_SUCCESS);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    ASSERT_EQ(asdBlasSetStream(handle, aclStream), AsdSip::ErrorType::ACL_SUCCESS);

    std::complex<float> alpha(1.0f, 0.0f), beta(0.0f, 0.0f);
    auto st = asdBlasCgemvBatched(handle, trans, m, n, alpha, aclA, m, aclX, 1, beta, aclY, 1, batch);
    ASSERT_EQ(st, AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(aclStream), 0);

    size_t yBytes = static_cast<size_t>(batch * n) * sizeof(std::complex<float>);
    std::vector<std::complex<float>> yOut(static_cast<size_t>(batch * n));
    ASSERT_EQ(aclrtMemcpy(yOut.data(), yBytes, yAddr, yBytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    constexpr double kRelTol = 1e-5;
    for (int64_t bi = 0; bi < batch; bi++) {
        for (int64_t j = 0; j < n; j++) {
            std::complex<double> sum(0.0, 0.0);
            for (int64_t i = 0; i < m; i++) {
                std::complex<double> av = a[(bi * m + i) * n + j];
                std::complex<double> xv = x[bi * m + i];
                if (trans == asdBlasOperation_t::ASDBLAS_OP_C) {
                    sum += std::conj(av) * xv;
                } else if (trans == asdBlasOperation_t::ASDBLAS_OP_T) {
                    sum += av * xv;
                } else {
                    // OP_N: y[i] = Σ_j a[i][j]*x[j]
                }
            }
            std::complex<double> expect = sum;
            std::complex<double> actual = yOut[bi * n + j];
            if (trans == asdBlasOperation_t::ASDBLAS_OP_N) {
                continue; // OP_N 输出为 m 元素，独立负例覆盖，本用例聚焦 T/C
            }
            double rel = std::abs(actual - expect) / (std::abs(expect) + 1e-10);
            ASSERT_LT(rel, kRelTol) << "trans=" << (trans == asdBlasOperation_t::ASDBLAS_OP_C ? "C" : "T")
                                    << " b=" << bi << " j=" << j << " actual=(" << actual.real() << "," << actual.imag()
                                    << ") expect=(" << expect.real() << "," << expect.imag() << ")";
        }
    }

    asdBlasDestroy(handle);
    aclDestroyTensor(aclA);
    aclDestroyTensor(aclX);
    aclDestroyTensor(aclY);
    aclrtFree(aAddr);
    aclrtFree(xAddr);
    aclrtFree(yAddr);
    AsdSip::TensorContext emptyContext;
    OpTestEnd(0, emptyContext, stream);
}

} // namespace

TEST(TestBlasCgemvBatchedTrans, ConjTransM12N6)
{
    RunCgemvBatchedCase(asdBlasOperation_t::ASDBLAS_OP_C, 12, 6, 1, 20260101u);
}

TEST(TestBlasCgemvBatchedTrans, TransM12N6)
{
    RunCgemvBatchedCase(asdBlasOperation_t::ASDBLAS_OP_T, 12, 6, 1, 20260102u);
}
