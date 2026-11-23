/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <securec.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/platform/platform_info.h>
#include "log/log.h"
#include "iamax.h"
#include "iamax_tiling_data.h"
#include "iamax_tiling.h"

constexpr int32_t MAXNUMF32ELEEACHCORE = 23040;      // 实数超过这么多个，需要多轮循环处理
constexpr int32_t MAX_NUM_COM_ELE_EACH_CORE = 11520; // 复数超过这么多个，需要多轮循环处理
constexpr int32_t BYTESPERBLOCK = 32;
constexpr int32_t BYTESPERREPEAT = 256;
constexpr int32_t F32LEN = 4;
constexpr uint32_t MAXVECTORNUM = 40;

namespace AsdSip {

using namespace Mki;

constexpr int32_t GM_RESULT_LEN = 2;
constexpr int32_t BYTE_LEN_4 = 4;
constexpr uint64_t ELEMENTS_IN_BLOCK = 8;
constexpr int32_t BLOCKS_PER_REPEAT = 8;
constexpr uint64_t MAX_COER_IN_REPEATE = 32;
constexpr int32_t SYS_WORK_SPACE = 16 * 1024 * 1024;
constexpr int32_t MAX_REPEATS = 255;

// 如果输入数据很大，UB需多次处理，并且轮次很多，不能等所有轮次都汇总完，这样中间结果太占内存，需要及时对部分中间结果取reduceMax
// 中间结果按2k规划，即2*1024/32=64次，即单核超过64次,就要取一次reduceMax，将空间占用降为一个block
// 64-63是因为有一个是历史压缩结果
constexpr int32_t DEAL_TIMES_EACH_CORE_REDUCE = 63;

uint32_t CeilA2B(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
};

uint32_t GetNeedVecCoreNum(uint32_t tensorLen, uint32_t elements)
{
    uint32_t needVecCoreNum = 1;
    if (tensorLen > MAXVECTORNUM * elements) {
        needVecCoreNum = MAXVECTORNUM;
    } else if (tensorLen > elements) {
        needVecCoreNum = tensorLen / elements; // 不足一个repeat的数据给第一个核，减少多核同步的可能性
    } else {
        needVecCoreNum = 1;
    }
    return needVecCoreNum;
}

AsdSip::AspbStatus IamaxTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    OpParam::Iamax param = AnyCast<OpParam::Iamax>(launchParam.GetParam());
    uint32_t numberElements = static_cast<uint32_t>(param.n);
    uint32_t dytpeFlag = static_cast<uint32_t>(param.dtype);

    uint32_t elementsPerRepeat = BYTESPERREPEAT / F32LEN;

    size_t sizeNum = launchParam.GetInTensor(0).desc.dims.size();
    uint32_t tensorLen = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[sizeNum - 1]);

    uint32_t complexNum = 2;
    if (dytpeFlag == 1) {
        tensorLen = tensorLen * complexNum;
        numberElements = numberElements * complexNum;
    }

    uint32_t minEleEachCore = elementsPerRepeat;
    uint32_t tmpLen = (tensorLen < numberElements) ? tensorLen : numberElements;
    uint32_t needVecCoreNum = GetNeedVecCoreNum(tmpLen, minEleEachCore);
    if (needVecCoreNum == 0) {
        needVecCoreNum = 1;
    }
    uint32_t rstLenAllCoreBytes = needVecCoreNum * GM_RESULT_LEN * BYTE_LEN_4;

    // 按repeat64均分，尽量保证每个核吃到整repeat的数据，尾块数据部分丢给头块核
    uint32_t minEleRepeatsNumber = tmpLen / minEleEachCore;
    uint32_t minEleRepeatTail = tmpLen % minEleEachCore;

    uint32_t minEleRepeatsNumberEachCore = minEleRepeatsNumber / needVecCoreNum;
    uint32_t minEleRepeatsNumbeTail = minEleRepeatsNumber % needVecCoreNum;

    uint32_t* allMem = nullptr;
    try {
        allMem = new uint32_t[12 * MAXVECTORNUM];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "allMem failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint32_t* startOffset = allMem;
    uint32_t* endOffset = allMem + MAXVECTORNUM;
    uint32_t* eleTotalEachCore = allMem + 2 * MAXVECTORNUM;
    uint32_t* dealLenEachTime = allMem + 3 * MAXVECTORNUM;
    uint32_t* dealTimesEachCore = allMem + 4 * MAXVECTORNUM;
    uint32_t* reduceMaxRstsLenEachCore = allMem + 5 * MAXVECTORNUM;
    uint32_t* dealLenUpBlockEachTime = allMem + 6 * MAXVECTORNUM;
    uint32_t* totalRptCntNor = allMem + 7 * MAXVECTORNUM;
    uint32_t* totalRptCntNorRemainder = allMem + 8 * MAXVECTORNUM;
    uint32_t* rptBatchCntNor = allMem + 9 * MAXVECTORNUM;
    uint32_t* rptBatchCntNorRemainder = allMem + 10 * MAXVECTORNUM;
    uint32_t* rmdRptLenNor = allMem + 11 * MAXVECTORNUM;

    uint32_t eleLenEachCore = 0;
    for (uint32_t i = 0; i < needVecCoreNum; i++) {
        eleLenEachCore = minEleRepeatsNumberEachCore * minEleEachCore;
        if (i == 0) {
            startOffset[i] = 0;
        } else {
            startOffset[i] = endOffset[i - 1];
        }

        // 均分给所有核
        if (minEleRepeatsNumbeTail > 0) {
            eleLenEachCore += minEleEachCore;
            minEleRepeatsNumbeTail--;
        }
        dealTimesEachCore[i] = 0;
        dealLenEachTime[i] = eleLenEachCore; // 不带尾块算
        if (eleLenEachCore > 0 && eleLenEachCore <= MAXNUMF32ELEEACHCORE) {
            dealTimesEachCore[i] = 1;
        } else if (eleLenEachCore > MAXNUMF32ELEEACHCORE) {
            dealTimesEachCore[i] = CeilA2B(eleLenEachCore, MAXNUMF32ELEEACHCORE);
            dealLenEachTime[i] = MAXNUMF32ELEEACHCORE;
        } else {
            dealTimesEachCore[i] = 0;
            dealLenEachTime[i] = 0;
        }

        uint32_t dealLenEachTimeAttachTail = dealLenEachTime[i];
        if (i == 0 && minEleRepeatTail != 0) {
            eleLenEachCore += minEleRepeatTail; // 尾块全给第一个核
            if (dealTimesEachCore[i] == 0) {
                dealTimesEachCore[i] = 1;
            }
            dealLenEachTimeAttachTail += minEleRepeatTail;
        }
        endOffset[i] = startOffset[i] + eleLenEachCore;
        eleTotalEachCore[i] = eleLenEachCore;

        // 默认就申请这么大
        reduceMaxRstsLenEachCore[i] = DEAL_TIMES_EACH_CORE_REDUCE * ELEMENTS_IN_BLOCK + ELEMENTS_IN_BLOCK;
        dealLenUpBlockEachTime[i] = CeilA2B(dealLenEachTimeAttachTail, ELEMENTS_IN_BLOCK) * ELEMENTS_IN_BLOCK;

        totalRptCntNor[i] = dealLenEachTime[i] / elementsPerRepeat;
        totalRptCntNorRemainder[i] = dealLenEachTime[i] % elementsPerRepeat; // should calc
        rptBatchCntNor[i] = totalRptCntNor[i] / MAX_REPEATS;                 // limit by L0 API, should calc
        rptBatchCntNorRemainder[i] = totalRptCntNor[i] % MAX_REPEATS;        // should calc
        rmdRptLenNor[i] = rptBatchCntNorRemainder[i] * elementsPerRepeat;
    }
    uint32_t maxRepeatLen = MAX_REPEATS * elementsPerRepeat;

    IamaxTilingData* tilingDataPtr = reinterpret_cast<AsdSip::IamaxTilingData*>(kernelInfo.GetTilingHostAddr());

    if (tilingDataPtr == nullptr) {
        delete[] allMem;
        allMem = nullptr;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    tilingDataPtr->needVecCoreNum = needVecCoreNum;
    tilingDataPtr->dytpeFlag = dytpeFlag;
    tilingDataPtr->rstLenAllCoreBytes = rstLenAllCoreBytes;
    tilingDataPtr->tailCount = minEleRepeatTail;
    tilingDataPtr->maxRepeatLen = maxRepeatLen;
    uint32_t copyLen = MAXVECTORNUM * sizeof(uint32_t);
    memcpy_s(tilingDataPtr->startOffset, copyLen, startOffset, copyLen);
    memcpy_s(tilingDataPtr->eleTotalEachCore, copyLen, eleTotalEachCore, copyLen);
    memcpy_s(tilingDataPtr->dealTimesEachCore, copyLen, dealTimesEachCore, copyLen);
    memcpy_s(tilingDataPtr->dealLenEachTime, copyLen, dealLenEachTime, copyLen);
    memcpy_s(tilingDataPtr->reduceMaxRstsLenEachCore, copyLen, reduceMaxRstsLenEachCore, copyLen);
    memcpy_s(tilingDataPtr->dealLenUpBlockEachTime, copyLen, dealLenUpBlockEachTime, copyLen);
    memcpy_s(tilingDataPtr->totalRptCntNor, copyLen, totalRptCntNor, copyLen);
    memcpy_s(tilingDataPtr->totalRptCntNorRemainder, copyLen, totalRptCntNorRemainder, copyLen);
    memcpy_s(tilingDataPtr->rptBatchCntNor, copyLen, rptBatchCntNor, copyLen);
    memcpy_s(tilingDataPtr->rptBatchCntNorRemainder, copyLen, rptBatchCntNorRemainder, copyLen);
    memcpy_s(tilingDataPtr->rmdRptLenNor, copyLen, rmdRptLenNor, copyLen);

    delete[] allMem;
    allMem = nullptr;

    // mix模式下，每个AICore会启动两个AIVcore,
    uint32_t mixCoreNum = CeilA2B(needVecCoreNum, 2);
    if (mixCoreNum == 0) {
        mixCoreNum = 1;
    }
    kernelInfo.SetBlockDim(mixCoreNum);

    kernelInfo.GetScratchSizes().push_back(SYS_WORK_SPACE + needVecCoreNum * GM_RESULT_LEN * BYTE_LEN_4);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
