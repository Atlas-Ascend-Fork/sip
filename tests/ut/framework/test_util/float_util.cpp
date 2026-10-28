/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "float_util.h"
#include "mki/utils/fp16/fp16_t.h"
#include <random>
#include "log/log.h"

bool FloatUtil::FloatJudgeEqual(float expected, float actual, float atol, float rtol)
{
    bool judge = std::abs(expected - actual) <= (atol + rtol * std::abs(actual));
    ASDSIP_LOG_IF(!judge, ERROR) << "judge float expected: " << expected << ", actual: " << actual;
    return judge;
}

template <class T>
void FloatUtil::GenerateCode(float lower, float upper, std::vector<T>& code)
{
    // 固定种子 + 工具链无关 LCG（issue #306），不用 std::random_device
    uint32_t state = 42U;
    for (size_t i = 0; i < code.size(); i++) {
        code[i] = static_cast<T>(ut_random::Uniform(state, lower, upper));
    }
}

template void FloatUtil::GenerateCode(float lower, float upper, std::vector<Mki::fp16_t>& code);
template void FloatUtil::GenerateCode(float lower, float upper, std::vector<float>& code);
