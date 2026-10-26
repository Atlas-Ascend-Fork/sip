
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
#include "sgetri_batched_tiling_data.h"
#include "sgetri_batched.h"
#include "sgetri_batched_tiling.h"

static constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;
static constexpr uint32_t BASE_BLOCK_ELENUM = 16;
static constexpr uint32_t TILE_ELENUM = 512;

namespace AsdSip {

AsdSip::AspbStatus SgetriBatchedTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    void* tilingData = kernelInfo.GetTilingHostAddr();
    SgetriBatchedTilingData* tilingDataPtr = (SgetriBatchedTilingData*)(tilingData);
    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    OpParam::SgetriBatched param = AnyCast<OpParam::SgetriBatched>(launchParam.GetParam());
    uint32_t dtype = static_cast<uint32_t>(param.dtype);
    uint32_t n = static_cast<uint32_t>(param.n);
    uint32_t batchSize = static_cast<uint32_t>(param.batchSize);

    ASDSIP_LOG(INFO) << "n:" << param.n;
    ASDSIP_LOG(INFO) << "batchSize:" << param.batchSize;

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    cubeCoreNum = (cubeCoreNum == 0) ? 1 : cubeCoreNum;

    tilingDataPtr->dtype = dtype;
    tilingDataPtr->n = n;
    tilingDataPtr->batchSize = batchSize;
    tilingDataPtr->blockM = BASE_BLOCK_ELENUM;
    tilingDataPtr->blockN = BASE_BLOCK_ELENUM;
    tilingDataPtr->tileM = TILE_ELENUM;

    kernelInfo.SetBlockDim(cubeCoreNum);
    kernelInfo.GetScratchSizes() = {WORKSPACE_SIZE};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
