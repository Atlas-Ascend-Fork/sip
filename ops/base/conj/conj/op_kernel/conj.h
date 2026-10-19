/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CONJ_N_D_H
#define CONJ_N_D_H

#include <type_traits>
#include "kernel_operator.h"

namespace Conj {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_BLOCK = 32;
constexpr int32_t BYTES_PER_REPEAT = 256;
constexpr int32_t MAX_CAST_COUNT = 512;
constexpr uint32_t MAX_DATA_COUNT = 8 * 1024;

class Conj {
public:
    __aicore__ inline Conj(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AsdSip::ConjTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t dataCount);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> inTensorsGM;
    GlobalTensor<float> outTensorsGM;

    int64_t blockIdx = 0;

    // tiling params
    uint32_t coreNum = 0;
    uint32_t dataNum = 0;
    uint32_t len = 0;
    uint32_t blockOffset = 0;
};

__aicore__ inline void Conj::Init(GM_ADDR x, GM_ADDR y, const AsdSip::ConjTilingData* tilingData)
{
    blockIdx = GetBlockIdx();

    inTensorsGM.SetGlobalBuffer((__gm__ float*)x);
    outTensorsGM.SetGlobalBuffer((__gm__ float*)y);

    pipe.InitBuffer(dataQueue, BUFFER_NUM, MAX_DATA_COUNT * sizeof(float));
    pipe.InitBuffer(outQueue, BUFFER_NUM, MAX_DATA_COUNT * sizeof(float));

    dataNum = tilingData->dataNum;
    coreNum = tilingData->coreNum;

    len = tilingData->len;
    blockOffset = len * blockIdx;
    if (blockIdx == coreNum - 1) {
        len = tilingData->tail;
    }
}

__aicore__ inline void Conj::Process()
{
    uint32_t times = len / MAX_DATA_COUNT;
    uint32_t reminder = len % MAX_DATA_COUNT;

    uint32_t offset = blockOffset;
    for (uint32_t i = 0; i < times; i++) {
        CopyIn(offset, MAX_DATA_COUNT);
        Compute(MAX_DATA_COUNT);
        CopyOut(offset, MAX_DATA_COUNT);
        offset += MAX_DATA_COUNT;
    }

    if (reminder > 0) {
        // 尾块按实际元素数搬运（issue #253）：原 CeilA2B(reminder, 8) * 8 向上取整
        // 后整块 DataCopy，tail 非 8 倍数时读/写均越界张量末尾最多 7 个 float；
        // 搬入改 DataCopyPad 按实际 reminder（pad 值 0 不影响有效数据，计算仍按
        // 对齐量执行），搬出同样按实际 reminder，不再越界
        uint32_t dataCount = CeilA2B(reminder, 8) * 8;
        LocalTensor<float> dataLocal = dataQueue.AllocTensor<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(reminder * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[offset], copyParams, padParams);
        dataQueue.EnQue(dataLocal);
        Compute(dataCount);
        LocalTensor<float> outLocal = outQueue.DeQue<float>();
        DataCopyPad(outTensorsGM[offset], outLocal, copyParams);
        outQueue.FreeTensor(outLocal);
    }
}

__aicore__ inline void Conj::CopyIn(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<float> dataLocal = dataQueue.AllocTensor<float>();
    DataCopy(dataLocal, inTensorsGM[offset], dataCount);
    dataQueue.EnQue(dataLocal);
}

__aicore__ inline void Conj::Compute(uint32_t dataCount)
{
    LocalTensor<float> dataLocal = dataQueue.DeQue<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

    uint64_t mask[2] = {6148914691236517205, 0};
    uint64_t mask_sub[2] = {__UINT64_C(12297829382473034410), 0};
    uint64_t repeatTimes = (dataCount * sizeof(float) + BYTES_PER_REPEAT - 1) / BYTES_PER_REPEAT;
    pipe_barrier(PIPE_V);
    Duplicate<float>(outLocal, 0, dataCount);
    pipe_barrier(PIPE_V);
    Copy(outLocal, dataLocal, mask, repeatTimes, {1, 1, 8, 8});
    pipe_barrier(PIPE_V);
    Sub(outLocal, outLocal, dataLocal, mask_sub, repeatTimes, {1, 1, 1, 8, 8, 8});
    pipe_barrier(PIPE_V);

    dataQueue.FreeTensor(dataLocal);
    outQueue.EnQue<float>(outLocal);
}

__aicore__ inline void Conj::CopyOut(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopy(outTensorsGM[offset], outLocal, dataCount);
    outQueue.FreeTensor(outLocal);
}

} // namespace Conj

#endif // CONJ_N_D_H
