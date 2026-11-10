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
#include <stdexcept>
#include <memory>
#include <mki/tensor.h>
// UT 未包含 core/fft/utils 的头文件搜索路径，按 fftmixcache 先例以相对路径引入
#include "../../../../../core/fft/utils/include/utils/fftensor_cache.h"

// FFTensorCache builder 空指针契约负向用例（issue #380）：getCoeff 对 func() 返回值
// 判空兜底，nullptr builder 触发 std::runtime_error 而非空指针解引用

TEST(TestFFTensorCacheNegative, TestBuilderNullptrRejected)
{
    AsdSip::CoeffKey key = {0U, 0U, {16}, true};
    EXPECT_THROW(FFTensorCache::getCoeff(key, []() -> AsdSip::FFTensor* { return nullptr; }), std::runtime_error);
}
