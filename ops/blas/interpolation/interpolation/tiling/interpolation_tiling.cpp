/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "interpolation_tiling_data.h"
#include "interpolation.h"
#include "tiling/tiling_api.h"
#include "interpolation_tiling.h"

#include <iostream>
#include <complex>

using namespace matmul_tiling;

namespace AsdSip {
using namespace Mki;

constexpr uint32_t BASE_MATMUL_SIZE = 16;
static constexpr size_t BASIC_BLOCK = 32;
constexpr uint64_t UB_DIVIDE_NUM = 8;
static constexpr size_t COMPUTE_PER_SIZE = 256;
static constexpr size_t COMPUTE_PER_NUM = 16; // 2个vector计算，pos是float : 2 * 32 / sizeof(float)
static constexpr uint32_t ALPHA = 4;
static constexpr uint32_t WORKSPACE_SIZE = (ALPHA * COMPUTE_PER_NUM * 2) * (COMPUTE_PER_NUM * 2) + 16 +
                                           (ALPHA * COMPUTE_PER_NUM + 8) * 2; // workspace计算
constexpr int32_t DEFAULT_CUBE_NUM = 20;

void InterpolationTCubeTiling(uint8_t* tilingData)
{
    matmul_tiling::MatmulApiTiling tilingApi;
    optiling::TCubeTiling tCubeTiling;

    uint32_t matSizeM = COMPUTE_PER_NUM * 2;
    uint32_t matSizeN = 1;
    uint32_t matSizeK = ALPHA * COMPUTE_PER_NUM * 2;

    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetShape(matSizeM, matSizeN, matSizeK);
    tilingApi.SetOrgShape(matSizeM, matSizeN, matSizeK);
    tilingApi.SetBufferSpace(-1, -1, -1);
    tilingApi.SetBias(false);

    int ret = tilingApi.GetTiling(tCubeTiling);
    if (ret == -1) {
        ASDSIP_LOG(ERROR) << "gen tiling failed";
    }
    uint32_t tilingSize = tCubeTiling.GetDataSize();
    tCubeTiling.SaveToBuffer(tilingData + sizeof(InterpolationTilingData), tilingSize);
}

AsdSip::AspbStatus InterpolationTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    uint8_t* tilingData = kernelInfo.GetTilingHostAddr();

    InterpolationTilingData* tilingDataPtr = reinterpret_cast<AsdSip::InterpolationTilingData*>(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    int32_t cubeCoreNum = static_cast<int32_t>(Mki::PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }
    cubeCoreNum = cubeCoreNum > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNum;

    AsdSip::OpParam::Interpolation param = AnyCast<OpParam::Interpolation>(launchParam.GetParam());

    // tab 布局约束（issue #458，与入口同口径）：内核 sincTab 打包布局按 16/32 固定排布
    ASDSIP_CHECK(param.tabInterpNum == 16 && param.tabQuantNum == 32,
                 "InterpolationTiling only supports tabInterpNum=16 and tabQuantNum=32 in current version.",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    // interpLength 约束（issue #457，与入口同口径）：内核按 COMPUTE_PER_NUM 整除分块，
    // 非 16 倍数尾块不进计算循环，输出残留旧值
    ASDSIP_CHECK(param.interpLength > 0 && param.interpLength % COMPUTE_PER_NUM == 0,
                 "InterpolationTiling get interpLength <= 0 or not a multiple of 16.",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    int32_t linesPerCore = 0;
    int32_t tailLines = 0;
    int32_t blocksPerLine = (param.interpLength + COMPUTE_PER_NUM - 1) / COMPUTE_PER_NUM;
    int32_t usedCubeCoreNum = cubeCoreNum;

    if (param.batch >= cubeCoreNum) {
        linesPerCore = param.batch / cubeCoreNum;
        tailLines = param.batch % cubeCoreNum;
    } else {
        linesPerCore = 0;
        tailLines = param.batch;
    }

    if (linesPerCore == 0 && blocksPerLine * tailLines < cubeCoreNum) {
        usedCubeCoreNum = blocksPerLine * tailLines;
    }

    // 值初始化整结构（issue #460，对齐已合入值初始化 CleanCode 口径）：reserved 保留
    // 字段随整结构下发设备，未初始化会携带宿主缓冲不确定值上行，赋值前整体清零
    *tilingDataPtr = {};

    tilingDataPtr->tabInterpNum = param.tabInterpNum;
    tilingDataPtr->tabQuantNum = param.tabQuantNum;
    tilingDataPtr->batch = param.batch;
    tilingDataPtr->signalLength = param.signalLength;
    tilingDataPtr->interpLength = param.interpLength;
    tilingDataPtr->numPerBlock = COMPUTE_PER_NUM;
    tilingDataPtr->linesPerCore = linesPerCore;
    tilingDataPtr->tailLines = tailLines;
    tilingDataPtr->blocksPerLine = blocksPerLine;
    tilingDataPtr->usedCubeCoreNum = usedCubeCoreNum;
    tilingDataPtr->workspaceSize = WORKSPACE_SIZE;

    ASDSIP_LOG(INFO) << "tabInterpNum : " << tilingDataPtr->tabInterpNum;
    ASDSIP_LOG(INFO) << "  tabQuantNum : " << tilingDataPtr->tabQuantNum;
    ASDSIP_LOG(INFO) << "  batch : " << tilingDataPtr->batch;
    ASDSIP_LOG(INFO) << "  signalLength : " << tilingDataPtr->signalLength;
    ASDSIP_LOG(INFO) << "  interpLength : " << tilingDataPtr->interpLength;
    ASDSIP_LOG(INFO) << "  numPerBlock : " << tilingDataPtr->numPerBlock;
    ASDSIP_LOG(INFO) << "  linesPerCore : " << tilingDataPtr->linesPerCore;
    ASDSIP_LOG(INFO) << "  tailLines : " << tilingDataPtr->tailLines;
    ASDSIP_LOG(INFO) << "  blocksPerLine : " << tilingDataPtr->blocksPerLine;
    ASDSIP_LOG(INFO) << "  usedCubeCoreNum : " << tilingDataPtr->usedCubeCoreNum;
    ASDSIP_LOG(INFO) << "  workspaceSize : " << tilingDataPtr->workspaceSize;
    InterpolationTCubeTiling(tilingData);

    uint64_t workspaceSize = WORKSPACE_SIZE * usedCubeCoreNum * sizeof(float) * 2;
    kernelInfo.SetBlockDim(usedCubeCoreNum);
    kernelInfo.GetScratchSizes() = {workspaceSize};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
