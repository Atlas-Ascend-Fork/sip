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

#include "test_util/util.cpp"

// 直接编入被测实现：FFTMixCache 为 sip_pta 侧纯 C++ 缓存逻辑，无独立测试链接产物，
// handle 创建/销毁经头文件 inline 实现走真实 asdFft 接口（与既有 FFT 用例同设备环境）
#include "../../../../../sip_pta/csrc/FFTMixCache.cpp"

using namespace AsdSip;

namespace {

op_api::FFTCacheKey Make1DCacheKey()
{
    op_api::FFTCacheKey key{};
    key.fftParam.fftXSize = 8;
    key.fftParam.fftYSize = 0;
    key.fftParam.fftZSize = 0;
    key.fftParam.fftType = asdFftType::ASCEND_FFT_C2C;
    key.fftParam.direction = asdFftDirection::ASCEND_FFT_FORWARD;
    key.fftParam.batchSize = 1;
    key.fftParam.dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
    return key;
}

} // namespace

// 空缓存 + capacity<=0 负向用例：get() 淘汰分支与 setCapacity() 收缩循环
// 带空表守卫后，不得对空 std::list 调用 front()/pop_front()（未定义行为），
// 守卫口径与姐妹实现 BlasMixCache 一致
TEST(TestFftMixCache, EmptyCacheNonPositiveCapacity)
{
    MkiRtStream stream = OpTestInit(0);

    op_api::FFTMixCache cache(0);
    EXPECT_EQ(cache.getCapacity(), 0);
    EXPECT_EQ(cache.getSize(), 0);

    op_api::FFTCacheKey key = Make1DCacheKey();
    // miss 路径：守卫跳过淘汰（capacity=0 且空表），创建真实句柄并正常入缓存
    op_api::FFTCacheValue value = cache.get(key);
    ASSERT_NE(value.handle, nullptr);
    EXPECT_EQ(cache.getSize(), 1);
    EXPECT_EQ(cache.getCapacity(), 0);

    // 负容量收缩：清空后 0 > -1 仍成立，无守卫时将对空表 pop_front（UB），守卫使循环安全终止
    cache.setCapacity(-1);
    EXPECT_EQ(cache.getCapacity(), -1);
    EXPECT_EQ(cache.getSize(), 0);

    // capacity<0 下再次 miss：守卫跳过淘汰，句柄重建与缓存写入不受影响
    op_api::FFTCacheValue valueAgain = cache.get(key);
    ASSERT_NE(valueAgain.handle, nullptr);
    EXPECT_EQ(cache.getSize(), 1);

    TensorContext emptyContext;
    OpTestEnd(0, emptyContext, stream);
}

// 收缩路径常规行为：setCapacity(0) 将已有条目安全清空后循环终止
TEST(TestFftMixCache, ShrinkToZeroClearsEntries)
{
    MkiRtStream stream = OpTestInit(0);

    op_api::FFTMixCache cache(2);
    op_api::FFTCacheKey key = Make1DCacheKey();
    ASSERT_NE(cache.get(key).handle, nullptr);
    ASSERT_EQ(cache.getSize(), 1);

    cache.setCapacity(0);
    EXPECT_EQ(cache.getCapacity(), 0);
    EXPECT_EQ(cache.getSize(), 0);

    TensorContext emptyContext;
    OpTestEnd(0, emptyContext, stream);
}
