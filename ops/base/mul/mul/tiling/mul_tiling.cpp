/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <securec.h>
#include "mul_tiling_data.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mul.h"
#include "mul_tiling.h"

namespace AsdSip {
using namespace Mki;

static constexpr int64_t COMPLEX64_PER_BLOCK = 4; // one block = 32B
static constexpr int64_t COMPLEX32_PER_BLOCK = 8; // one block = 32B

// Set tiling data value
static void SetTilingData(CMulTilingData& tilingDataPtr, CMulTilingData tilingData, int64_t vecCoreNum)
{
    tilingDataPtr.n = tilingData.n;
    tilingDataPtr.useCoreNum = tilingData.useCoreNum;

    auto ret = memcpy_s(tilingDataPtr.startOffset, sizeof(tilingDataPtr.startOffset), tilingData.startOffset,
                        vecCoreNum * sizeof(int64_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "startOffset memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ret = memcpy_s(tilingDataPtr.calNum, sizeof(tilingDataPtr.calNum), tilingData.calNum, vecCoreNum * sizeof(int64_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "calNum memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
}

// Calculate tiling data value
static CMulTilingData CalTilingData(int64_t totalEleNum, int64_t vecCoreNum, int64_t complexPerBlock)
{
    CMulTilingData tilingData;
    tilingData.n = totalEleNum;
    tilingData.useCoreNum = 0;
    if (vecCoreNum == 0) {
        ASDSIP_LOG(ERROR) << "vecCoreNum invalid: vecCoreNum = " << vecCoreNum;
        vecCoreNum = 1;
    }
    // Set zero for startOffset and calNum
    for (int64_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }
    // Num of blocks
    int64_t totalBlockNum = totalEleNum / complexPerBlock;
    // remain row num
    int64_t remainNum = totalEleNum % complexPerBlock;

    if (totalBlockNum == 0) {
        // Use only 1 AIV core.
        tilingData.calNum[0] = remainNum;
    } else if (totalBlockNum < vecCoreNum) {
        for (int64_t i = 0; i < totalBlockNum; i++) {
            tilingData.startOffset[i] = complexPerBlock * i;
            tilingData.calNum[i] = complexPerBlock;
        }
        tilingData.startOffset[totalBlockNum] = complexPerBlock * totalBlockNum;
        tilingData.calNum[totalBlockNum] += remainNum;
        tilingData.useCoreNum = totalBlockNum + 1;
    } else {
        int64_t procNumEachCore;
        int64_t remainProc;

        procNumEachCore = totalBlockNum / vecCoreNum;
        remainProc = totalBlockNum % vecCoreNum;

        int64_t currOffset = 0;
        int64_t currCalNum = 0;

        for (int64_t i = 0; i < vecCoreNum; i++) {
            if (i < remainProc) {
                currCalNum = (procNumEachCore + 1) * complexPerBlock;
            } else {
                currCalNum = procNumEachCore * complexPerBlock;
            }
            tilingData.startOffset[i] = currOffset;
            tilingData.calNum[i] = currCalNum;
            currOffset += currCalNum;
        }
        tilingData.calNum[vecCoreNum - 1] += remainNum;
        tilingData.useCoreNum = vecCoreNum;
    }
    return tilingData;
}

static int64_t ConfigCMulTilingData(CMulTilingData* tilingDataPtr, int64_t totalEleNum, int64_t vecCoreNum,
                                    int64_t complexPerBlock)
{
    CMulTilingData tilingData = CalTilingData(totalEleNum, vecCoreNum, complexPerBlock);
    SetTilingData(*tilingDataPtr, tilingData, vecCoreNum);
    return tilingData.useCoreNum;
}

AsdSip::AspbStatus CMulTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    void* tilingData = kernelInfo.GetTilingHostAddr();
    CMulTilingData* tilingDataPtr = reinterpret_cast<AsdSip::CMulTilingData*>(tilingData);
    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    int64_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    // 上界钳位（issue #291）：CMulTilingData 的 calNum/startOffset 为栈上定长
    // [48] 数组，vecCoreNum 超 48 时越界写破坏宿主栈；同仓 core/blas 批量 tiling
    // 族均对此钳位，此处补齐。当前在售平台核数 <= 48，钳位不改变现有行为
    constexpr int64_t CMUL_MAX_VEC_CORE = 48;
    if (vecCoreNum > CMUL_MAX_VEC_CORE) {
        ASDSIP_LOG(WARN) << "vecCoreNum " << vecCoreNum << " exceeds CMulTilingData capacity 48, clamped.";
        vecCoreNum = CMUL_MAX_VEC_CORE;
    }
    OpParam::CMul param = AnyCast<OpParam::CMul>(launchParam.GetParam());
    int64_t complexPerBlock = (param.cMulType == OpParam::CMul::CMulType::MUL_C64) ? COMPLEX64_PER_BLOCK :
                                                                                     COMPLEX32_PER_BLOCK;
    int64_t useCoreNum = ConfigCMulTilingData(tilingDataPtr, param.n, vecCoreNum, complexPerBlock);
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    uint64_t workspaceSize = 0;
    kernelInfo.SetBlockDim(useCoreNum);
    kernelInfo.GetScratchSizes() = {workspaceSize};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
