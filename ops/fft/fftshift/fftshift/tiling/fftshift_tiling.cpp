/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/utils/status/status.h"
#include "mki/utils/platform/platform_info.h"
#include "mki/utils/SVector/SVector.h"
#include "utils/assert.h"
#include "log/log.h"
#include "tiling_data.h"
#include "fftshift.h"
#include "fftshift_tiling.h"

namespace AsdSip {

using Mki::AnyCast;
using Mki::CoreType;
using Mki::KernelInfo;
using Mki::LaunchParam;
using Mki::PlatformInfo;
using Mki::PlatformType;
using Mki::Status;

AsdSip::AspbStatus FftShiftTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    const auto& param = AnyCast<OpParam::FftShift>(launchParam.GetParam());

    ASDSIP_CHECK(param.row > 0, "FftShift row must be positive", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_CHECK(param.col > 0, "FftShift col must be positive", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_CHECK(param.batch > 0, "FftShift batch must be positive", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    int64_t sliceSize = param.row * param.col;
    int64_t dist = (param.dist > 0) ? param.dist : sliceSize;
    ASDSIP_CHECK(dist >= sliceSize, "FftShift dist must not be smaller than row * col",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    FftShiftTilingData* tilingDataPtr = reinterpret_cast<AsdSip::FftShiftTilingData*>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    tilingDataPtr->row = param.row;
    tilingDataPtr->col = param.col;
    tilingDataPtr->batch = param.batch;
    tilingDataPtr->dist = dist;
    tilingDataPtr->totalRows = param.batch * param.row;
    // Dynamic UB size from the platform (e.g. 248KB on 950PR, 192KB on 910B):
    // the kernel uses this to size its staging buffers instead of a hardcoded
    // conservative budget, utilizing the full per-core UB capacity.
    tilingDataPtr->ubSize = static_cast<int64_t>(PlatformInfo::Instance().GetUbSize());
    if (tilingDataPtr->ubSize <= 0) {
        tilingDataPtr->ubSize = 236 * 1024; // conservative fallback
    }

    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    if (maxCore == 0) {
        maxCore = 1;
    }
    uint32_t needCoreNum = (tilingDataPtr->totalRows > static_cast<int64_t>(maxCore)) ?
                               maxCore :
                               static_cast<uint32_t>(tilingDataPtr->totalRows);
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }

    kernelInfo.SetBlockDim(needCoreNum);
    fprintf(stderr, "[FFTSHIFT-UB] ubSize=%lld bytes\n", (long long)(tilingDataPtr->ubSize));
    kernelInfo.GetScratchSizes().push_back(0);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
