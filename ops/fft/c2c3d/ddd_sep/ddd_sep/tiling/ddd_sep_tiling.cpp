/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>

#include <mki/utils/status/status.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "tiling_data.h"
#include "ddd.h"
#include "ddd_sep_tiling.h"

namespace AsdSip {
using namespace Mki;
AsdSip::AspbStatus DddSepTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    const auto& param = AnyCast<OpParam::Ddd>(launchParam.GetParam());

    DddSepTilingData* tilingDataPtr = reinterpret_cast<AsdSip::DddSepTilingData*>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    tilingDataPtr->batchSize = param.batchSize;
    tilingDataPtr->fftX = param.fftX;
    tilingDataPtr->fftY = param.fftY;
    tilingDataPtr->fftZ = param.fftZ;

    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));

    // batchSize * fftX * fftY * fftZ * 2
    // workspace 字节数提升为 uint64_t 并做溢出检查（issue #263，对齐 2D
    // dd_tiling.cpp 口径）：原 uint32_t 在 batch×X×Y×Z×2×sizeof(float) ≥ 2^32
    // 时按模截断（batch=4、512^3 恰截为 0），scratch 分配远小于内核实际读写量
    // 造成设备内存越界写；超限配置显式拒绝
    uint64_t totalFloats = 0;
    uint64_t tmp = 0;
    if (__builtin_mul_overflow(static_cast<uint64_t>(param.batchSize), static_cast<uint64_t>(param.fftX), &tmp) ||
        __builtin_mul_overflow(tmp, static_cast<uint64_t>(param.fftY), &tmp) ||
        __builtin_mul_overflow(tmp, static_cast<uint64_t>(param.fftZ), &tmp) ||
        __builtin_mul_overflow(tmp, static_cast<uint64_t>(2) * sizeof(float), &totalFloats)) {
        ASDSIP_LOG(ERROR) << "DddSep workspace size overflows uint64.";
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    // scratch 容量按框架接口以 uint32 承载（issue #263）：总字节数超过 UINT32_MAX
    // 时显式拒绝，杜绝截断后按小容量分配而内核全量写入的越界
    if (totalFloats > static_cast<uint64_t>(UINT32_MAX)) {
        ASDSIP_LOG(ERROR) << "DddSep workspace bytes exceed UINT32_MAX (" << totalFloats << "), unsupported.";
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    kernelInfo.SetBlockDim(maxCore);
    kernelInfo.GetScratchSizes().push_back(static_cast<uint32_t>(totalFloats));
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}

} // namespace AsdSip
