/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BuildGemmDftTensor(仓内构建器)与示例 BuildRotationEmbeddedPadded 的位级对拍;
// 纯宿主不碰 NPU;基准三文件拷贝自示例目录随编——对拍对象是真示例代码

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "utils/fft_gemm_coeffs.h"

// 数值基准(示例真实现,已拷贝至本目录)
#include "fft_coeffs.h"

namespace {

constexpr int32_t kL50 = 50;
constexpr int32_t kStride50 = 104; // RoundUp(2*50, 8) = RoundUp(100, 8),与示例 WS_STRIDE 同规则

// 与构建器同规则的行距:2L 向上取整到 8 的倍数
int32_t StrideOf(int32_t fftLen)
{
    const int32_t dim = fftLen * 2;
    return (dim + 7) / 8 * 8;
}

// 析构说明:~FFTensor 定义在 fftensor_cache.o(尚未被任何 .so 链入,第 3 步接线后
// 随整链进入)。测试用空删除器持有,避免引用该符号;进程退出即回收,单份 41.6KB。
struct KeepTensor {
    void operator()(AsdSip::FFTensor*) const {}
};

// 生成 [2L, stride] 矩阵对:ref = 示例基准,got = 仓内构建器输出(独占持有)
std::unique_ptr<AsdSip::FFTensor, KeepTensor> BuildPair(int32_t fftLen, bool forward, std::vector<float>& ref)
{
    const int32_t stride = StrideOf(fftLen);
    ref.assign(static_cast<size_t>(fftLen) * 2 * stride, 0.0f);
    Fft::BuildRotationEmbeddedPadded(ref.data(), fftLen, stride, forward ? -1 : 1);
    return std::unique_ptr<AsdSip::FFTensor, KeepTensor>(AsdSip::BuildGemmDftTensor(fftLen, forward));
}

// 用例 1a:位级对拍,L=50 正向
TEST(TestFftGemmCoeffs, BitwiseMatchL50Forward)
{
    std::vector<float> ref;
    auto got = BuildPair(kL50, true, ref);
    ASSERT_NE(got, nullptr);
    ASSERT_NE(got->hostData, nullptr);
    ASSERT_EQ(got->dataSize, ref.size() * sizeof(float));
    EXPECT_EQ(std::memcmp(got->hostData, ref.data(), ref.size() * sizeof(float)), 0)
        << "L50 forward: 仓内构建器与示例基准存在位级差异";
}

// 用例 1b:位级对拍,L=50 逆向
TEST(TestFftGemmCoeffs, BitwiseMatchL50Inverse)
{
    std::vector<float> ref;
    auto got = BuildPair(kL50, false, ref);
    ASSERT_NE(got, nullptr);
    ASSERT_NE(got->hostData, nullptr);
    ASSERT_EQ(got->dataSize, ref.size() * sizeof(float));
    EXPECT_EQ(std::memcmp(got->hostData, ref.data(), ref.size() * sizeof(float)), 0)
        << "L50 inverse: 仓内构建器与示例基准存在位级差异";
}

// 用例 2:DC 行与列。W[p][q] = e^{2πi·pq/L},p=0 或 q=0 时恒为 1:
// 行 0 偶数列全 1、奇数列全 0;列 0 偶数行全 1、奇数行全 0
TEST(TestFftGemmCoeffs, DcRowAndColumn)
{
    std::vector<float> ref;
    auto got = BuildPair(kL50, true, ref);
    ASSERT_NE(got, nullptr);
    const float* w = static_cast<const float*>(got->hostData);
    const int32_t stride = kStride50;
    for (int32_t q = 0; q < kL50; ++q) {
        EXPECT_FLOAT_EQ(w[0 * stride + 2 * q], 1.0f) << "row0 even col q=" << q;
        EXPECT_FLOAT_EQ(w[0 * stride + 2 * q + 1], 0.0f) << "row0 odd col q=" << q;
    }
    for (int32_t p = 0; p < kL50; ++p) {
        EXPECT_FLOAT_EQ(w[(2 * p) * stride + 0], 1.0f) << "col0 even row p=" << p;
        EXPECT_FLOAT_EQ(w[(2 * p + 1) * stride + 0], 0.0f) << "col0 odd row p=" << p;
    }
}

// 用例 3:2×2 块模长。块 [[cosθ, dir·sinθ], [−dir·sinθ, cosθ]] 的两行都应满足
// cos² + sin² = 1(float 存储后的误差上界约 2.4e-7,取 1e-6 容差)。
TEST(TestFftGemmCoeffs, DiagonalBlockUnitNorm)
{
    std::vector<float> ref;
    auto got = BuildPair(kL50, true, ref);
    ASSERT_NE(got, nullptr);
    const float* w = static_cast<const float*>(got->hostData);
    const int32_t stride = kStride50;
    for (int32_t p = 1; p < kL50; ++p) {
        for (int32_t q = 1; q < kL50; ++q) {
            const float re0 = w[(2 * p) * stride + 2 * q];
            const float im0 = w[(2 * p) * stride + 2 * q + 1];
            EXPECT_NEAR(re0 * re0 + im0 * im0, 1.0f, 1e-6f) << "row 2p, p=" << p << " q=" << q;
            const float re1 = w[(2 * p + 1) * stride + 2 * q];
            const float im1 = w[(2 * p + 1) * stride + 2 * q + 1];
            EXPECT_NEAR(re1 * re1 + im1 * im1, 1.0f, 1e-6f) << "row 2p+1, p=" << p << " q=" << q;
        }
    }
}

// 用例 4:参数化泛化——L=8(无垫列)、L=26(有垫列:52→56)、L=64(2 幂,无垫列),
// 两方向各对拍一次。提前验证任意 L 的正确性,不等第 5 步 A/B。
TEST(TestFftGemmCoeffs, BitwiseMatchOtherLengths)
{
    for (int32_t fftLen : {8, 26, 64}) {
        for (bool forward : {true, false}) {
            std::vector<float> ref;
            auto got = BuildPair(fftLen, forward, ref);
            ASSERT_NE(got, nullptr);
            ASSERT_NE(got->hostData, nullptr);
            ASSERT_EQ(got->dataSize, ref.size() * sizeof(float));
            EXPECT_EQ(std::memcmp(got->hostData, ref.data(), ref.size() * sizeof(float)), 0)
                << "fftLen=" << fftLen << " forward=" << forward;
        }
    }
}

} // namespace
