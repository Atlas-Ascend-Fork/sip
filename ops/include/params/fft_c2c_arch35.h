/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDSIP_PARAMS_FFT_C2C_ARCH35_H
#define ASDSIP_PARAMS_FFT_C2C_ARCH35_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AsdSip {

struct FftC2CArch35StageTilingData {
    int64_t batch;
    int64_t n;
    int64_t totalButterflies;
    int32_t nHalf;
    int32_t len;
    int32_t half;
    int32_t logNhalf;
    int32_t logHalf;
    int32_t twOffset;
    int32_t outer;
    int32_t isInverse;
    int32_t scaleOut;
    int32_t transpose;
};

namespace OpParam {

/**
 * Parameter struct for FftC2C Arch35 (Ascend 950) operation.
 *
 * Fields:
 *   fftN         - Full signal length
 *   batchSize    - Number of batched FFTs
 *   radixListLen - Number of radix decomposition stages (length of plan)
 *   isInverse    - 0 for forward FFT, 1 for inverse FFT
 *   isMixedRadix - false for radix-2 kernel path, true for mixed-radix kernel path
 */
struct FftC2CArch35 {
    int64_t fftN;
    int64_t batchSize;
    int64_t radixListLen;
    int32_t isInverse;
    bool isMixedRadix;
    bool isGemm = false; // 第 4 步:true=GEMM 短维 pass(2 入张量)或收尾转置 pass(1 入张量,fftM 经 radixListLen 携带)
    bool isNFused = false; // N 维融合 pass:2 入张量(数据+ex89 twiddle 表),fftM 经 radixListLen
                           // 携带
    bool isHybrid = false; // 混合核:2 入张量(数据+W 垫宽),radixListLen 携带 2D fftN(瓦片行数),fftN 字段携带 2D
                           // fftM

    bool operator==(const FftC2CArch35& other) const
    {
        return this->fftN == other.fftN && this->batchSize == other.batchSize &&
               this->radixListLen == other.radixListLen && this->isInverse == other.isInverse &&
               this->isMixedRadix == other.isMixedRadix && this->isGemm == other.isGemm &&
               this->isNFused == other.isNFused && this->isHybrid == other.isHybrid;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: FftC2CArch35"
           << ", fftN:" << fftN << ", batchSize:" << batchSize << ", radixListLen:" << radixListLen
           << ", isInverse:" << isInverse << ", isMixedRadix:" << isMixedRadix << ", isGemm:" << isGemm
           << ", isNFused:" << isNFused << ", isHybrid:" << isHybrid;
        return ss.str();
    }
};

} // namespace OpParam
} // namespace AsdSip

#endif
