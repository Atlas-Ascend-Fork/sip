/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "conj_tiling.h"
#include "tiling_data.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"

namespace AsdSip {
using namespace Mki;

static constexpr uint32_t FLOAT_NUM_PER_BLOCK = 8; // one block = 32B
AsdSip::AspbStatus ConjTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    if (maxCore == 0) {
        maxCore = 1;
    }
    // 元素数全程 int64 计算（issue #346）：原 static_cast<uint32_t>(Numel) 后乘 2，
    // Numel>=2^31 时在 uint32 域回绕为 0/错值，tiling 数据整体错误
    const int64_t numel = launchParam.GetInTensor(0).Numel();
    ASDSIP_CHECK(numel > 0, "ConjTiling get empty tensor (Numel <= 0).",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    const int64_t size64 = numel * 2;
    // ConjTilingData.dataNum 为 uint32 位宽，超界显式拒绝防静默截断
    ASDSIP_CHECK(size64 <= static_cast<int64_t>(UINT32_MAX), "ConjTiling get tensor too large (2 * Numel > 2^32-1).",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    uint32_t size = static_cast<uint32_t>(size64);
    uint32_t len = (size / maxCore + FLOAT_NUM_PER_BLOCK - 1) / FLOAT_NUM_PER_BLOCK * FLOAT_NUM_PER_BLOCK;
    uint32_t seqLenLowerBound = 64;
    if (len < seqLenLowerBound) {
        len = seqLenLowerBound;
    }
    uint32_t needCoreNum = (size + len - 1) / len;
    if (needCoreNum == 0) {
        // 下限钳制（issue #346）：size > 0 时数学上不触发，防御性消除
        // tail = size - len * (needCoreNum - 1) 的无符号下溢路径
        needCoreNum = 1;
    }
    uint32_t tail = size - len * (needCoreNum - 1);

    ConjTilingData* tilingDataPtr = reinterpret_cast<AsdSip::ConjTilingData*>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    tilingDataPtr->coreNum = needCoreNum;
    tilingDataPtr->dataNum = size;
    tilingDataPtr->len = len;
    tilingDataPtr->tail = tail;

    kernelInfo.SetBlockDim(needCoreNum);
    kernelInfo.GetScratchSizes().push_back(0);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
