
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

#include "utils/assert.h"
#include "log/log.h"
#include "mki/types.h"
#include "mki/utils/platform/platform_info.h"
#include "cgemm_batched.h"
#include "cgemm_batched_tiling_data.h"
#include "cgemm_batched_tiling.h"

namespace AsdSip {
using namespace Mki;

constexpr int64_t UB_SIZE = 192 * 1024;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t GATHER_OFFSET_SIZE = 1024;

static int64_t CeilDiv(int64_t x, int64_t base) { return (x + base - 1) / base; }

AsdSip::AspbStatus CgemmBatchedTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    auto dtype = launchParam.GetInTensor(0).desc.dtype;
    int64_t complexSize = static_cast<int64_t>(GetTensorElementSize(dtype));

    int64_t maxCubeCores = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (maxCubeCores == 0) {
        maxCubeCores = 1;
    }

    OpParam::CgemmBatched param = AnyCast<OpParam::CgemmBatched>(launchParam.GetParam());

    int64_t m = param.m;
    int64_t k = param.k;
    int64_t n = param.n;
    int64_t batch = param.batch;

    int64_t useCubeCores = batch < maxCubeCores ? batch : maxCubeCores;

    int64_t remainUb = UB_SIZE - GATHER_OFFSET_SIZE * static_cast<int64_t>(sizeof(uint32_t));
    int64_t rightSize = k * n;

    int64_t subBatch = CeilDiv(batch, useCubeCores);
    // there are 2 buffers
    // there are 2 inter-spaces
    int64_t miniBatch = remainUb / NUM_TWO / NUM_TWO / (rightSize * complexSize);

    void* tilingData = kernelInfo.GetTilingHostAddr();
    CgemmBatchedTilingData* tilingDataPtr = reinterpret_cast<CgemmBatchedTilingData*>(tilingData);
    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    tilingDataPtr->m = m;
    tilingDataPtr->k = k;
    tilingDataPtr->n = n;
    tilingDataPtr->batch = batch;
    tilingDataPtr->subBatch = subBatch;
    tilingDataPtr->miniBatch = miniBatch;
    ASDSIP_LOG(DEBUG) << "tilingData: "
                      << "OpName: CgemmBatched"
                      << ", m:" << m << ", k:" << k << ", n:" << n << ", batch:" << batch << ", subBatch:" << subBatch
                      << ", miniBatch:" << miniBatch;

    // there are 2 vectors
    // there are 2 buffers
    uint64_t workspaceSize = static_cast<uint64_t>(NUM_TWO * NUM_TWO * miniBatch * rightSize * complexSize *
                                                   useCubeCores);
    kernelInfo.SetBlockDim(useCubeCores);
    kernelInfo.GetScratchSizes() = {workspaceSize};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
