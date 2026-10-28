/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLOATUTIL_H
#define FLOATUTIL_H

#include <cstdint>
#include <vector>
#include "log/log.h"
#include "mki/utils/status/status.h"

// 工具链无关的确定性随机数（issue #306）：std::default_random_engine 与
// uniform_real_distribution 的输出序列是实现定义的，同一 seed 在不同 libstdc++
// 上产生不同数据，导致 UT 输入随编译工具链漂移——部分 blas/solver 用例在特定
// 工具链下随机到极端组合（累加 inf/nan 或抵消型精度超限）且无法在他机复现。
// 改用固定参数 LCG（Numerical Recipes），任意编译器/平台序列完全一致。
namespace ut_random {
inline uint32_t Next(uint32_t& state)
{
    state = state * 1664525U + 1013904223U;
    return state;
}

// 均匀分布 [lower, upper)
inline float Uniform(uint32_t& state, float lower, float upper)
{
    constexpr float kInvU32 = 1.0f / 4294967296.0f; // 2^-32
    return lower + (static_cast<float>(Next(state)) * kInvU32) * (upper - lower);
}
} // namespace ut_random

class FloatUtil {
public:
    /**
     * @brief 指定精度下判断两float数据是否相同
     * @param excepted      期望数据
     * @param actual        真实计算数据
     * @param atol          绝对误差的阈值，例如0.001
     * @param rtol          相对误差的阈值，例如0.001
     * @return 数据相同返回true，不同返回false
     */
    static bool FloatJudgeEqual(float expected, float actual, float atol, float rtol);

    /**
     * @brief 生成指定大小范围的float类型随机数
     * @param lower         随机数大小下限
     * @param upper         随机数大小上限
     * @param code          输出，生成的随机数vector，数据类型转为T
     */
    template <class T>
    static void GenerateCode(float lower, float upper, std::vector<T>& code);
};

#endif
