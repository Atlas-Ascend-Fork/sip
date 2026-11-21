/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <complex>

#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "interpbycoeff.h"
#include "interpbycoeff_tiling_data.h"
#include "interpbycoeff_tiling.h"

using namespace matmul_tiling;

namespace AsdSip {
using namespace Mki;

constexpr int32_t REPEAT_32 = 64;
constexpr int32_t REPEAT_64 = 32;
constexpr int32_t BASE_MATMUL_SIZE = 16;
constexpr int32_t L0_SIZE = 65536;

void LogTiling(InterpByCoeffTilingData tilingDataPtr)
{
    ASDSIP_LOG(INFO) << "  batch : " << tilingDataPtr.batch;
    ASDSIP_LOG(INFO) << "  rsNum : " << tilingDataPtr.rsNum;
    ASDSIP_LOG(INFO) << "  totalSubcarrier : " << tilingDataPtr.totalSubcarrier;
    ASDSIP_LOG(INFO) << "  interpLength : " << tilingDataPtr.interpLength;
    ASDSIP_LOG(INFO) << "  batchPerCore : " << tilingDataPtr.batchPerCore;
    ASDSIP_LOG(INFO) << "  tailBatch : " << tilingDataPtr.tailBatch;
    ASDSIP_LOG(INFO) << "  splitN : " << tilingDataPtr.splitN;
    ASDSIP_LOG(INFO) << "  splitLength : " << tilingDataPtr.splitLength;
    ASDSIP_LOG(INFO) << "  blocksPerBatch : " << tilingDataPtr.blocksPerBatch;
    ASDSIP_LOG(INFO) << "  blockLength : " << tilingDataPtr.blockLength;
    ASDSIP_LOG(INFO) << "  usedCubeCoreNum : " << tilingDataPtr.usedCubeCoreNum;
    ASDSIP_LOG(INFO) << "  workspaceSize : " << tilingDataPtr.workspaceSize;
    ASDSIP_LOG(INFO) << "  dataType : " << tilingDataPtr.dataType;
}

bool InterpByCoeffTCubeTiling(uint8_t* tilingData, int32_t tilingKey)
{
    matmul_tiling::MatmulApiTiling tilingApi;
    optiling::TCubeTiling tCubeTiling;

    InterpByCoeffTilingData* tilingDataPtr = reinterpret_cast<AsdSip::InterpByCoeffTilingData*>(tilingData);
    int32_t matSizeM = tilingDataPtr->interpLength;
    int32_t matSizeN = tilingDataPtr->totalSubcarrier * 2;
    int32_t matSizeK = tilingDataPtr->rsNum * 2;

    if (tilingKey == 1) {
        tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    } else {
        tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    }
    tilingApi.SetShape(matSizeM, matSizeN, matSizeK);
    tilingApi.SetOrgShape(matSizeM, matSizeN, matSizeK);
    tilingApi.SetBufferSpace(-1, -1, -1);
    tilingApi.SetBias(false);

    int ret = tilingApi.GetTiling(tCubeTiling);
    if (ret == -1) {
        ASDSIP_LOG(ERROR) << "gen tiling failed";
        return false;
    }
    uint32_t tilingSize = tCubeTiling.GetDataSize();
    tCubeTiling.SaveToBuffer(tilingData + sizeof(InterpByCoeffTilingData), tilingSize);
    return true;
}

AsdSip::AspbStatus InterpByCoeffTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    uint8_t* tilingData = kernelInfo.GetTilingHostAddr();
    InterpByCoeffTilingData* tilingDataPtr = reinterpret_cast<AsdSip::InterpByCoeffTilingData*>(tilingData);
    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    int32_t cubeCoreNum = static_cast<int32_t>(Mki::PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }

    TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
    AsdSip::OpParam::InterpByCoeff param = AnyCast<OpParam::InterpByCoeff>(launchParam.GetParam());
    int32_t tilingKey = 1; // Complex32
    int32_t repeatNum = REPEAT_32;
    int32_t maxBaseN = static_cast<int32_t>(L0_SIZE / BASE_MATMUL_SIZE / sizeof(float)) / (param.rsNum * 2) *
                       BASE_MATMUL_SIZE; // Complex32
    ASDSIP_CHECK(maxBaseN != 0, "maxBaseN is 0!", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    if (inDtype == TensorDType::TENSOR_DTYPE_COMPLEX64) {
        tilingKey = 0;
        repeatNum = REPEAT_64;
        maxBaseN = static_cast<int32_t>(L0_SIZE / BASE_MATMUL_SIZE / (sizeof(float) * 2)) / (param.rsNum * 2) *
                   BASE_MATMUL_SIZE;
    }
    int32_t splitN = (param.totalSubcarrier + maxBaseN - 1) / maxBaseN;
    int32_t splitLength = (param.totalSubcarrier / splitN + repeatNum - 1) / repeatNum * repeatNum;

    int32_t batchPerCore = 0;
    int32_t tailBatch = 0;
    int32_t blocksPerBatch = 0;
    int32_t blockLength = 0;
    int32_t usedCubeCoreNum = cubeCoreNum;
    if (param.batch >= cubeCoreNum) {
        batchPerCore = param.batch / cubeCoreNum;
        tailBatch = param.batch % cubeCoreNum;
    } else {
        batchPerCore = 0;
        tailBatch = param.batch;
    }

    if (tailBatch != 0) {
        if (cubeCoreNum % tailBatch == 0) {
            blocksPerBatch = cubeCoreNum / tailBatch;
        } else {
            blockLength = (param.totalSubcarrier * tailBatch + cubeCoreNum - 1) / cubeCoreNum;
            blocksPerBatch = (param.totalSubcarrier + blockLength - 1) / blockLength;
        }
        blockLength = (param.totalSubcarrier + blocksPerBatch - 1) / blocksPerBatch;
        if (blockLength > splitLength) {
            blockLength = splitLength;
        } else {
            blockLength = (blockLength + repeatNum - 1) / repeatNum * repeatNum;
        }
        blocksPerBatch = (param.totalSubcarrier + blockLength - 1) / blockLength;
    }

    if (batchPerCore == 0 && blocksPerBatch * tailBatch < cubeCoreNum) {
        usedCubeCoreNum = blocksPerBatch * tailBatch;
    }

    // 值初始化整结构（issue #460，对齐已合入值初始化 CleanCode 口径）：reserved 保留
    // 字段随整结构下发设备，未初始化会携带宿主缓冲不确定值上行，赋值前整体清零
    *tilingDataPtr = {};

    tilingDataPtr->batch = param.batch;
    tilingDataPtr->rsNum = param.rsNum;
    tilingDataPtr->totalSubcarrier = param.totalSubcarrier;
    tilingDataPtr->interpLength = param.interpLength;
    tilingDataPtr->batchPerCore = batchPerCore;
    tilingDataPtr->tailBatch = tailBatch;
    tilingDataPtr->splitN = splitN;
    tilingDataPtr->splitLength = splitLength;
    tilingDataPtr->blocksPerBatch = blocksPerBatch;
    tilingDataPtr->blockLength = blockLength;
    tilingDataPtr->usedCubeCoreNum = usedCubeCoreNum;
    tilingDataPtr->workspaceSize = L0_SIZE;
    tilingDataPtr->dataType = tilingKey;
    LogTiling(*tilingDataPtr);
    auto res = InterpByCoeffTCubeTiling(tilingData, tilingKey);
    ASDSIP_CHECK(res, "Failed to execute func InterpByCoeffTCubeTiling!",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint64_t workspaceSize = static_cast<uint64_t>(L0_SIZE) * static_cast<uint64_t>(usedCubeCoreNum) * 2;
    kernelInfo.SetBlockDim(usedCubeCoreNum);
    kernelInfo.SetTilingId(tilingKey);
    kernelInfo.GetScratchSizes() = {workspaceSize};

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
