/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SWAPLAST2AXES_AIV_H
#define SWAPLAST2AXES_AIV_H

#include <type_traits>
#include "kernel_operator.h"
#include "../../tiling/swap_last_2axes_tiling_data.h"

using namespace AscendC;

namespace SwapLast2Axes {

constexpr uint64_t K_BUFFER_NUM = 2;

constexpr uint64_t K_N_FLOATS_PER_BLOCK = 8;

constexpr uint64_t K_PART_ROW = 192;
constexpr uint64_t K_PART_COL = 16;
constexpr uint64_t K_PART_BYTES = K_PART_ROW * K_PART_COL * K_N_FLOATS_PER_BLOCK;
constexpr uint64_t K_BLOCK_BYTES = 32;

template <typename T>
class SwapLast2AxesAIV {
public:
    __aicore__ inline SwapLast2AxesAIV(){};
    __aicore__ inline void Init(GM_ADDR inTensor, GM_ADDR outTensor, AsdSip::SwapLast2AxesTilingData* tilingdata,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t partRow, uint64_t partCol, uint64_t inOffset);
    __aicore__ inline void ComputeTransDataTo5HD(uint64_t partRow, uint64_t partCol);
    __aicore__ inline void ComputeGather(uint64_t partRow, uint64_t partCol);
    __aicore__ inline void CopyOut(uint64_t partRow, uint64_t partCol, uint64_t outOffset);

    // tiling params

    TPipe pipe;
    TQue<QuePosition::VECIN, K_BUFFER_NUM> inQueue;
    TQue<QuePosition::VECIN, K_BUFFER_NUM> interQueue;
    TQue<QuePosition::VECOUT, K_BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorGM;
    GlobalTensor<T> outTensorGM;

    TBuf<TPosition::VECCALC> offsetBuf;

    LocalTensor<uint32_t> offsetLocal;

    uint64_t batch;
    uint64_t row;
    uint64_t col;

    uint64_t cores;
};

template <>
__aicore__ inline void SwapLast2AxesAIV<float>::Init(GM_ADDR inTensor, GM_ADDR outTensor,
                                                     AsdSip::SwapLast2AxesTilingData* tilingdata, TPipe* pipe)
{
    cores = AscendC::GetBlockNum();

    inTensorGM.SetGlobalBuffer((__gm__ float*)inTensor);
    outTensorGM.SetGlobalBuffer((__gm__ float*)outTensor);

    batch = tilingdata->batch;
    row = tilingdata->row;
    col = tilingdata->col;

    pipe->InitBuffer(inQueue, K_BUFFER_NUM, K_PART_BYTES);
    pipe->InitBuffer(interQueue, K_BUFFER_NUM, K_PART_BYTES);
    pipe->InitBuffer(outQueue, K_BUFFER_NUM, K_PART_BYTES);

    pipe->InitBuffer(offsetBuf, 64 * sizeof(float));
    offsetLocal = offsetBuf.Get<uint32_t>();
    for (uint64_t i = 0; i < 32; i += 1) {
        offsetLocal.SetValue(i * 2, sizeof(float) * i);
    }

    for (uint64_t i = 0; i < 32; i += 1) {
        offsetLocal.SetValue(i * 2 + 1, K_PART_ROW * sizeof(float) + sizeof(float) * i);
    }
}

template <>
__aicore__ inline void SwapLast2AxesAIV<float>::CopyIn(uint64_t partRow, uint64_t partCol, uint64_t inOffset)
{
    LocalTensor<float> inLocal = inQueue.AllocTensor<float>();

    if (col % 4 == 0) {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = partRow;
        dataCopyParams.blockLen = partCol / 4;
        dataCopyParams.srcStride = (col - partCol) / 4;
        dataCopyParams.dstStride = (K_PART_COL - partCol) / 4;

        DataCopy(inLocal, inTensorGM[inOffset], dataCopyParams);
    } else {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = partRow;
        dataCopyExtParams.blockLen = partCol * 8;
        dataCopyExtParams.srcStride = (col - partCol) * 8;
        dataCopyExtParams.dstStride = (K_PART_COL - partCol) / 4;

        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;

        DataCopyPad(inLocal, inTensorGM[inOffset], dataCopyExtParams, dataCopyPadExtParams);
    }

    inQueue.EnQue<float>(inLocal);
}

template <>
__aicore__ inline void SwapLast2AxesAIV<float>::ComputeTransDataTo5HD(uint64_t partRow, uint64_t partCol)
{
    LocalTensor<float> inLocal = inQueue.DeQue<float>();
    LocalTensor<float> interLocal = interQueue.AllocTensor<float>();

    uint64_t repeats_in_row = (partRow + 16 - 1) / 16;
    uint64_t repeats_in_col = (partCol * 2 + 8 - 1) / 8;

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (repeats_in_row == 1) {
        transDataParams.repeatTimes = 1;
        transDataParams.dstRepStride = 0;
        transDataParams.srcRepStride = 0;
    } else {
        transDataParams.repeatTimes = repeats_in_row;
        transDataParams.dstRepStride = 16 / K_N_FLOATS_PER_BLOCK;
        transDataParams.srcRepStride = 16 * (K_PART_COL * 2 / K_N_FLOATS_PER_BLOCK);
    }

    for (uint64_t j = 0; j < repeats_in_col; j++) {
        uint64_t srcLocalList[16];
        uint64_t dstLocalList[16];

        for (uint64_t i = 0; i < 16; i++) {
            srcLocalList[i] = (uint64_t)(inLocal[j * 8 + K_PART_COL * 2 * i].GetPhyAddr());
        }

        for (uint64_t i = 0; i < 16; i++) {
            dstLocalList[i] = (uint64_t)(interLocal[j * 192 * 8 + (i / 2) * K_PART_ROW + (i % 2) * K_N_FLOATS_PER_BLOCK]
                                             .GetPhyAddr());
        }

        TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParams);
    }

    interQueue.EnQue(interLocal);
    inQueue.FreeTensor(inLocal);
}

template <>
__aicore__ inline void SwapLast2AxesAIV<float>::ComputeGather(uint64_t partRow, uint64_t partCol)
{
    LocalTensor<float> interLocal = interQueue.DeQue<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

    for (uint64_t i = 0; i < partCol; i++) {
        for (uint64_t j = 0; j < ((partRow * 2 + 64 - 1) / 64); j++) {
            Gather(outLocal[i * K_PART_ROW * 2 + 64 * j], interLocal[i * K_PART_ROW * 2 + 32 * j], offsetLocal, 0, 64);
        }
    }

    outQueue.EnQue(outLocal);
    interQueue.FreeTensor(interLocal);
}

template <>
__aicore__ inline void SwapLast2AxesAIV<float>::CopyOut(uint64_t partRow, uint64_t partCol, uint64_t outOffset)
{
    LocalTensor<float> outLocal = outQueue.DeQue<float>();

    if (row % 4 == 0) {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = partCol;
        dataCopyParams.blockLen = partRow / 4;
        dataCopyParams.srcStride = (K_PART_ROW - partRow) / 4;
        dataCopyParams.dstStride = (row - partRow) / 4;

        DataCopy(outTensorGM[outOffset], outLocal, dataCopyParams);
    } else {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = partCol;
        dataCopyExtParams.blockLen = partRow * 8;
        dataCopyExtParams.srcStride = (K_PART_ROW - partRow) / 4;
        dataCopyExtParams.dstStride = (row - partRow) * 8;

        DataCopyPad(outTensorGM[outOffset], outLocal, dataCopyExtParams);
    }

    outQueue.FreeTensor<float>(outLocal);
}

template <>
__aicore__ inline void SwapLast2AxesAIV<float>::Process()
{
    uint64_t blockIdx = GetBlockIdx();

    uint64_t partIdx = 0;
    uint64_t partsInRow = (row + SwapLast2Axes::K_PART_ROW - 1) / SwapLast2Axes::K_PART_ROW;
    uint64_t partsInCol = (col + SwapLast2Axes::K_PART_COL - 1) / SwapLast2Axes::K_PART_COL;

    for (uint64_t batchIdx = 0; batchIdx < batch; batchIdx++) {
        for (uint64_t rowIdx = 0; rowIdx < partsInRow; rowIdx++) {
            for (uint64_t colIdx = 0; colIdx < partsInCol; colIdx++) {
                if (partIdx % cores == blockIdx) {
                    uint64_t partRow = K_PART_ROW;
                    if (rowIdx == partsInRow - 1 && row % K_PART_ROW != 0) {
                        partRow = row % K_PART_ROW;
                    }

                    uint64_t partCol = K_PART_COL;
                    if (colIdx == partsInCol - 1 && col % K_PART_COL != 0) {
                        partCol = col % K_PART_COL;
                    }

                    uint64_t inOffset = batchIdx * row * col * 2 + rowIdx * K_PART_ROW * col * 2 +
                                        colIdx * K_PART_COL * 2;
                    uint64_t outOffset = batchIdx * row * col * 2 + colIdx * K_PART_COL * row * 2 +
                                         rowIdx * K_PART_ROW * 2;

                    CopyIn(partRow, partCol, inOffset);
                    ComputeTransDataTo5HD(partRow, partCol);
                    ComputeGather(partRow, partCol);
                    CopyOut(partRow, partCol, outOffset);
                }

                partIdx++;
            }
        }
    }
}

} // namespace SwapLast2Axes

#endif // SWAPLAST2AXES_AIV_H
