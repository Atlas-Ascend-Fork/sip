/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft_c2c_arch35_gemm_tiling.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"

namespace AsdSip {
using namespace Mki;

AspbStatus FftC2CArch35GemmTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        FftGemmTilingData* tilingDataPtr = reinterpret_cast<FftGemmTilingData*>(kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "gemm tilingDataPtr should not be empty",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        ASDSIP_CHECK(param.isGemm, "FftC2CArch35GemmTiling requires gemm mode",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        tilingDataPtr->rows = param.batchSize;
        tilingDataPtr->kDim = static_cast<int32_t>(param.fftN) * 2;
        tilingDataPtr->nDim = static_cast<int32_t>(param.fftN) * 2; // 无垫列:C 即标准布局
        tilingDataPtr->isInverse = param.isInverse;

        uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
        if (maxCore == 0) {
            maxCore = 1;
        }
        tilingDataPtr->coreNum = maxCore;
        kernelInfo.SetBlockDim(maxCore);
    }

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus FftC2CArch35TransTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        FftTransTilingData* tilingDataPtr = reinterpret_cast<FftTransTilingData*>(kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "trans tilingDataPtr should not be empty",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        ASDSIP_CHECK(param.isGemm, "FftC2CArch35TransTiling requires gemm mode",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        // 参数打包:fftN=真 fftN,batchSize=batch,radixListLen=fftM(脚手架约定)
        tilingDataPtr->batch = param.batchSize;
        tilingDataPtr->fftN = static_cast<int32_t>(param.fftN);
        tilingDataPtr->fftM = static_cast<int32_t>(param.radixListLen);

        uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
        if (maxCore == 0) {
            maxCore = 1;
        }
        kernelInfo.SetBlockDim(maxCore);
    }

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus FftC2CArch35NFusedTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        FftNFusedTilingData* tilingDataPtr = reinterpret_cast<FftNFusedTilingData*>(kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "nfused tilingDataPtr should not be empty",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        ASDSIP_CHECK(param.isNFused, "FftC2CArch35NFusedTiling requires nfused mode",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        // 实例库校验:当前唯一复合实例 (20,25) 服务 500=2²·5³
        ASDSIP_CHECK(param.fftN == 500, "nfused instantiation (20,25) serves fftN=500 only",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        ASDSIP_CHECK(param.radixListLen >= 1 && param.radixListLen <= 52, "nfused tile budget exceeded",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        tilingDataPtr->batch = param.batchSize;
        tilingDataPtr->fftM = static_cast<int32_t>(param.radixListLen);
        tilingDataPtr->direction = param.isInverse ? -1 : 1;
        tilingDataPtr->twFloatOff = param.isInverse ? 1040 : 0;

        uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
        if (maxCore == 0) {
            maxCore = 1;
        }
        tilingDataPtr->coreNum = static_cast<int32_t>(maxCore);
        kernelInfo.SetBlockDim(maxCore);
        // UB 窗口申报(字节):瓦片 fftM×1000 浮点 + twiddle 表 1040 浮点;经 RunWithArgs override 下发
        kernelInfo.SetLocalMemorySize(static_cast<uint32_t>(param.radixListLen) * 1000 * 4 + 1040 * 4);
    }

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus FftC2CArch35HybridTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        FftHybridTilingData* tilingDataPtr = reinterpret_cast<FftHybridTilingData*>(kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "hybrid tilingDataPtr should not be empty",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        ASDSIP_CHECK(param.isHybrid, "FftC2CArch35HybridTiling requires hybrid mode",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        // 参数打包(脚手架约定):fftN=2D fftM(短维 L),batchSize=rows,radixListLen=2D fftN(瓦片行数)
        const int64_t fftMLocal = param.fftN;               // 短维 L
        const int64_t fftNLocal = param.radixListLen;       // 长维(瓦片行数)
        const int64_t nDimLocal = (fftMLocal * 2 + 7) & ~7; // UB 行距 32B 对齐
        tilingDataPtr->batch = param.batchSize / fftNLocal; // rows → 矩阵数(内核按 [fftN,fftM] 矩阵寻址)
        tilingDataPtr->fftN = static_cast<int32_t>(fftNLocal);
        tilingDataPtr->fftM = static_cast<int32_t>(fftMLocal);
        tilingDataPtr->nDim = static_cast<int32_t>(nDimLocal);
        tilingDataPtr->isInverse = param.isInverse;
        const char* dbgEnv = std::getenv("ASDSIP_HYBRID_DEBUG");
        tilingDataPtr->debugMode = dbgEnv != nullptr ? std::atoi(dbgEnv) : 0;

        uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
        if (maxCore == 0) {
            maxCore = 1;
        }
        tilingDataPtr->coreNum = maxCore;
        kernelInfo.SetBlockDim(maxCore);
        // UB 窗口申报:瓦片 fftN×nDim 浮点 + 预留 1040 浮点(的 twiddle 表);经 RunWithArgs override 下发
        constexpr uint64_t UB_WINDOW_BYTES = 216ULL * 1024ULL;
        ASDSIP_CHECK(static_cast<uint64_t>(fftNLocal) * nDimLocal * 4ULL + 1040ULL * 4ULL <= UB_WINDOW_BYTES,
                     "hybrid UB tile budget exceeded", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        kernelInfo.SetLocalMemorySize(static_cast<uint32_t>(UB_WINDOW_BYTES));
    }

    return AsdSip::ErrorType::ACL_SUCCESS;
}

} // namespace AsdSip
