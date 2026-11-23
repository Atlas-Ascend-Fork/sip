/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_OPS_IAMAX_TILING_DATA
#define ASDSIP_OPS_IAMAX_TILING_DATA

#include <cstdint>
namespace AsdSip {

constexpr int32_t MAX_VECTOT = 40;

// tilingDatas似乎有上限，286*4
struct IamaxTilingData {
    uint32_t needVecCoreNum;
    uint32_t dytpeFlag;
    uint32_t rstLenAllCoreBytes;
    uint32_t tailCount;
    uint32_t maxRepeatLen;
    uint32_t startOffset[MAX_VECTOT];
    uint32_t eleTotalEachCore[MAX_VECTOT];
    uint32_t dealTimesEachCore[MAX_VECTOT];
    uint32_t dealLenEachTime[MAX_VECTOT];
    uint32_t reduceMaxRstsLenEachCore[MAX_VECTOT];
    uint32_t dealLenUpBlockEachTime[MAX_VECTOT];
    uint32_t totalRptCntNor[MAX_VECTOT];
    uint32_t totalRptCntNorRemainder[MAX_VECTOT]; // should calc
    uint32_t rptBatchCntNor[MAX_VECTOT];          // limit by L0 API, should calc
    uint32_t rptBatchCntNorRemainder[MAX_VECTOT]; // should calc
    uint32_t rmdRptLenNor[MAX_VECTOT];
};
} // namespace AsdSip
#endif
