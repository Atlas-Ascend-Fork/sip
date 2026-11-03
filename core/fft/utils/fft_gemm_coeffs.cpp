/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/fft_gemm_coeffs.h"

#include <cmath>

namespace AsdSip {
namespace {
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = K_PI * 2;
} // namespace

FFTensor* BuildGemmDftTensor(int64_t fftLen, bool forward, int64_t stride)
{
    const int32_t dim = static_cast<int32_t>(fftLen) * 2;
    if (stride <= 0) {
        stride = (dim + 7) / 8 * 8; // 默认:输出列 8 浮点对齐(L=50 时 100→104)
    }
    const double dir = forward ? -1.0 : 1.0;

    float* host = new float[static_cast<size_t>(dim) * stride]();
    for (int32_t p = 0; p < static_cast<int32_t>(fftLen); ++p) {
        for (int32_t q = 0; q < static_cast<int32_t>(fftLen); ++q) {
            // 角度与示例逐运算同序(位级对拍前提)
            const double angle = K_2PI * static_cast<double>(p * q) / static_cast<double>(fftLen);
            const double cosVal = std::cos(angle);
            const double sinVal = std::sin(angle);

            host[(2 * p) * stride + (2 * q)] = static_cast<float>(cosVal);
            host[(2 * p) * stride + (2 * q + 1)] = static_cast<float>(dir * sinVal);
            host[(2 * p + 1) * stride + (2 * q)] = static_cast<float>(-dir * sinVal);
            host[(2 * p + 1) * stride + (2 * q + 1)] = static_cast<float>(cosVal);
        }
    }

    FFTensor* tensor = new FFTensor;
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT,
                    Mki::TensorFormat::TENSOR_FORMAT_ND,
                    {static_cast<int64_t>(dim) * stride},
                    {},
                    0};
    tensor->hostData = host;
    tensor->dataSize = static_cast<size_t>(dim) * stride * sizeof(float);
    return tensor;
}

FFTensor* BuildFusedNTwiddleTensor()
{
    // ex89 布局(与内核 NF_*_OFF 一致):[0,1040) forward(sign=-1)/[1040,2080) inverse(sign=+1)
    constexpr int32_t kR4T1 = 125;
    constexpr int32_t kR4T2 = 125;
    constexpr int32_t kR4T3 = 125;
    constexpr int32_t kTw3 = 100;
    constexpr int32_t kTw4 = 20;
    constexpr int32_t kPerDir = 1040;

    float* host = new float[2 * kPerDir]();
    auto put = [host](int32_t floatOff, int32_t cidx, double angle) {
        host[floatOff + 2 * cidx] = static_cast<float>(std::cos(angle));
        host[floatOff + 2 * cidx + 1] = static_cast<float>(std::sin(angle));
    };
    auto buildOne = [&put](int32_t base, double sign) {
        for (int32_t j = 0; j < kR4T1; ++j) {
            put(base + 0, j, sign * K_2PI * j / 500.0);
        }
        for (int32_t j = 0; j < kR4T2; ++j) {
            put(base + 250, j, sign * K_2PI * j / 250.0);
        }
        for (int32_t j = 0; j < kR4T3; ++j) {
            put(base + 500, j, sign * K_2PI * 3 * j / 500.0);
        }
        for (int32_t j = 0; j < kTw3 / 4; ++j) {
            for (int32_t t = 1; t < 5; ++t) {
                put(base + 750, j * 4 + (t - 1), sign * K_2PI * j * t / 125.0);
            }
        }
        for (int32_t j = 0; j < kTw4 / 4; ++j) {
            for (int32_t t = 1; t < 5; ++t) {
                put(base + 950, j * 4 + (t - 1), sign * K_2PI * j * t / 25.0);
            }
        }
        for (int32_t s = 0; s < 5; ++s) {
            for (int32_t t = 0; t < 5; ++t) {
                put(base + 990, s * 5 + t, sign * K_2PI * s * t / 5.0);
            }
        }
    };
    buildOne(0, -1.0);
    buildOne(kPerDir, 1.0);

    FFTensor* tensor = new FFTensor;
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {2 * kPerDir}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = 2 * kPerDir * sizeof(float);
    return tensor;
}

FFTensor* BuildHybridCoeffTensor(int64_t fftLen, bool forward, int64_t stride)
{
    // 混合核 coeff:[W 垫宽 dim×stride | ex89 双向 tw 2080]。
    // 段内填充与 BuildGemmDftTensor/BuildFusedNTwiddleTensor 逐式一致(位级对拍前提)。
    const int32_t dim = static_cast<int32_t>(fftLen) * 2;
    if (stride <= 0) {
        stride = (dim + 7) / 8 * 8;
    }
    const int64_t wFloats = static_cast<int64_t>(dim) * stride;
    constexpr int32_t kPerDir = 1040;
    const int64_t total = wFloats + 2 * kPerDir;
    const double dir = forward ? -1.0 : 1.0;

    float* host = new float[static_cast<size_t>(total)]();
    for (int32_t p = 0; p < static_cast<int32_t>(fftLen); ++p) {
        for (int32_t q = 0; q < static_cast<int32_t>(fftLen); ++q) {
            const double angle = K_2PI * static_cast<double>(p * q) / static_cast<double>(fftLen);
            host[(2 * p) * stride + (2 * q)] = static_cast<float>(std::cos(angle));
            host[(2 * p) * stride + (2 * q + 1)] = static_cast<float>(dir * std::sin(angle));
            host[(2 * p + 1) * stride + (2 * q)] = static_cast<float>(-dir * std::sin(angle));
            host[(2 * p + 1) * stride + (2 * q + 1)] = static_cast<float>(std::cos(angle));
        }
    }
    auto put = [host, wFloats](int32_t floatOff, int32_t cidx, double angle) {
        host[wFloats + floatOff + 2 * cidx] = static_cast<float>(std::cos(angle));
        host[wFloats + floatOff + 2 * cidx + 1] = static_cast<float>(std::sin(angle));
    };
    auto buildOne = [&put](int32_t base, double sign) {
        for (int32_t j = 0; j < 125; ++j) {
            put(base + 0, j, sign * K_2PI * j / 500.0);
        }
        for (int32_t j = 0; j < 125; ++j) {
            put(base + 250, j, sign * K_2PI * j / 250.0);
        }
        for (int32_t j = 0; j < 125; ++j) {
            put(base + 500, j, sign * K_2PI * 3 * j / 500.0);
        }
        for (int32_t j = 0; j < 25; ++j) {
            for (int32_t t = 1; t < 5; ++t) {
                put(base + 750, j * 4 + (t - 1), sign * K_2PI * j * t / 125.0);
            }
        }
        for (int32_t j = 0; j < 5; ++j) {
            for (int32_t t = 1; t < 5; ++t) {
                put(base + 950, j * 4 + (t - 1), sign * K_2PI * j * t / 25.0);
            }
        }
        for (int32_t s = 0; s < 5; ++s) {
            for (int32_t t = 0; t < 5; ++t) {
                put(base + 990, s * 5 + t, sign * K_2PI * s * t / 5.0);
            }
        }
    };
    buildOne(0, -1.0);
    buildOne(kPerDir, 1.0);

    FFTensor* tensor = new FFTensor;
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {total}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = static_cast<size_t>(total) * sizeof(float);
    return tensor;
}

} // namespace AsdSip
