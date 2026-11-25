/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_FFT_ALL_MIX_H
#define ASDSIP_PARAMS_FFT_ALL_MIX_H

#include <string>
#include <sstream>
#include "mki/types.h"
#include "mki/utils/compare/compare.h"

namespace AsdSip {
namespace OpParam {
// Coefficients and device tiling must decompose each stage in the same order.
inline int32_t Arch35RealRadix(int64_t remaining)
{
    for (int32_t radix : {2, 3, 5, 7, 11, 13}) {
        if (remaining % radix == 0)
            return radix;
    }
    return 0;
}

struct FftAllMix {
    int parity;
    int64_t N_doing;
    int aiv_split_way;
    int64_t batchSize;
    int64_t fftN;
    size_t radixListLen;
    // for now, c2r is only inverse and inverse = 1, r2c is only forward and inverse = 0
    int32_t isInverse;
    int64_t workspace_input_size;
    int64_t workspace_output_size;
    int64_t workspace_sync_size;
    int64_t workspace_c2c_output_size;
    int64_t workspace_auxil_size;

    bool operator==(const FftAllMix& other) const
    {
        return this->parity == other.parity && this->N_doing == other.N_doing &&
               this->aiv_split_way == other.aiv_split_way && this->batchSize == other.batchSize &&
               this->fftN == other.fftN && this->radixListLen == other.radixListLen &&
               this->isInverse == other.isInverse && this->workspace_input_size == other.workspace_input_size &&
               this->workspace_output_size == other.workspace_output_size &&
               this->workspace_sync_size == other.workspace_sync_size &&
               this->workspace_c2c_output_size == other.workspace_c2c_output_size &&
               this->workspace_auxil_size == other.workspace_auxil_size;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: FftAllMix"
           << ", parity:" << parity << ", N_doing:" << N_doing << ", aiv_split_way:" << aiv_split_way
           << ", batchSize:" << batchSize << ", fftN:" << fftN << ", radixListLen:" << radixListLen
           << ", isInverse:" << isInverse << ", workspace_input_size:" << workspace_input_size
           << ", workspace_output_size:" << workspace_output_size << ", workspace_sync_size:" << workspace_sync_size
           << ", workspace_c2c_output_size:" << workspace_c2c_output_size
           << ", workspace_auxil_size:" << workspace_auxil_size;
        return ss.str();
    }
};
} // namespace OpParam
} // namespace AsdSip

#endif
