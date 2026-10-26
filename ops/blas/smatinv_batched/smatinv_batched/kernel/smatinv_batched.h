/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SMATINV_BATCHED_AIV_H
#define SMATINV_BATCHED_AIV_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace SmatinvBatched {

constexpr uint32_t BYTENUM_BLOCK = 32;
constexpr uint32_t BYTENUM_REPEAT = 256;

constexpr uint32_t MAX_CORE_CNT = 40;

// fp32: 8 float per 32B block; fp16: 16 half per 32B block
constexpr uint32_t ELENUM_BLOCK_FP32 = 8;
constexpr uint32_t ELENUM_REPEAT_FP32 = 64;
constexpr uint32_t ELENUM_BLOCK_FP16 = 16;
constexpr uint32_t ELENUM_REPEAT_FP16 = 128;

// one repeat processes 32KB of FP32 matrix data (single plane, real matrices)
constexpr uint32_t BASE_SIZE_ONE_ITERATION = 32 * 1024;                     // 32 kb
constexpr uint32_t BASE_FP32_NUM = BASE_SIZE_ONE_ITERATION / sizeof(float); // 8192
constexpr uint32_t MAX_BATCH_NUM = 128;
constexpr uint32_t MAX_STRIDE_NUM = 255;
constexpr uint32_t MAX_REPEAT_NUM = 255;
constexpr uint32_t MAX_UBUF_SIZE = 192 * 1024; // 192kb

/**
 * @brief Batched real-matrix inversion (Gauss-Jordan elimination, single plane FP32 compute).
 *
 * Supported dtypes: float (dtype==0), half (dtype==1). half input is cast to FP32 in UB,
 * computed in FP32, and cast back on output, mirroring the CmatinvBatched precision policy.
 *
 * UB layout per repeat (batchNumPerRepeat matrices, n x alignedN each, FP32):
 *   matARealLocal  | 32KB |  input matrices (after cast for half)
 *   invRealLocal   | 32KB |  running inverse (starts as identity)
 *   matRealLocal   | 32KB |  working copy of A (destroyed by elimination)
 *   + temp buffers (scalar collect/broadcast/update temps)
 */
template <typename T>
class SmatinvBatchedAIV {
public:
    __aicore__ inline SmatinvBatchedAIV(){};
    __aicore__ inline void Init(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dAinv, GM_ADDR workSpace, GM_ADDR tilingGm);

    __aicore__ inline void InitUbuf();
    __aicore__ inline void InitTempUbuf(LocalTensor<float> ubufLocal, uint32_t allocOffset);
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);

    __aicore__ inline void Process();
    __aicore__ inline void SingleProcess();

    __aicore__ inline void CopyMatBatchGmToUb(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t matColSize,
                                              uint32_t batchSize);
    __aicore__ inline void CopyUniMatBatchGmToUb(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t matColSize,
                                                 uint32_t batchSize);
    __aicore__ inline void CopyVecGmToUbUint32(LocalTensor<uint32_t> dst, GlobalTensor<uint32_t> src, uint32_t len);
    __aicore__ inline void CopyVecGmToUb(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t len);
    __aicore__ inline void CopyVecUbToGm(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t len);

    __aicore__ inline void ComputeScalarOffset();
    __aicore__ inline void PrepareDataForHalf(uint32_t matColSize, uint32_t batchSize);
    __aicore__ inline void CollectScalarSingle(LocalTensor<float> collectDst, uint32_t sOffset);
    __aicore__ inline void ScalarDiv();
    __aicore__ inline void BrcbScalar(LocalTensor<float> brcbDst, LocalTensor<float> scalarSrc);
    __aicore__ inline void RealScalarVecBatchMul(LocalTensor<float> dst, LocalTensor<float> src0,
                                                 LocalTensor<float> src1, uint32_t oneMatOffset);
    __aicore__ inline void RealScalarVecBatchSub(LocalTensor<float> dst, LocalTensor<float> src, uint32_t oneMatOffset);
    __aicore__ inline void PlaneRowVecScalarBatchDiv(LocalTensor<float> dst, LocalTensor<float> src1,
                                                     uint32_t oneMatOffset);
    __aicore__ inline void ComputeAndUpdateAj(LocalTensor<float> Aj, uint32_t oneMatOffset);
    __aicore__ inline void Copyout(uint32_t outOffset, uint32_t batchSize);

private:
    static constexpr bool IS_FLOAT = IsSameType<T, float>::value; // half or float

    TPipe pipe;

    GlobalTensor<T> inAGM;
    GlobalTensor<float> uniRealGM;
    GlobalTensor<T> outGM;
    GlobalTensor<float> workGM;

    TBuf<QuePosition::VECCALC> uBuf;

    // main buf
    LocalTensor<float> matARealLocal; // input A (fp32 view after optional cast)
    LocalTensor<float> invRealLocal;  // running inverse
    LocalTensor<float> matRealLocal;  // working copy of A (destroyed)

    // temp buf
    LocalTensor<uint32_t> collectScalarOffsetLocal;
    LocalTensor<float> preLocal;        // pivot a_ii per batch
    LocalTensor<float> preBrcbLocal;    // broadcasted pivot a_ii per row
    LocalTensor<float> afterLocal;      // a_ji per batch
    LocalTensor<float> resLocal;        // factor = a_ji / a_ii per batch
    LocalTensor<float> scalarBrcbLocal; // broadcasted factor per row
    LocalTensor<float> updateTempLocal; // factor * A[i,:] temp

    // tiling
    uint32_t n = 0;
    uint32_t alignedMatSize = 0; // aligned n (fp32 element per row)
    uint32_t batchNumPerRepeat = 0;
    uint32_t oneMatUbOffsetFp32 = 0; // n * alignedMatSize
    uint32_t offset = 0;             // global element offset of this core
    uint32_t calNum = 0;             // matrix num of this core
    uint32_t coreIdx = 0;
    bool isFp32Input = true;
};

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::Init(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dAinv, GM_ADDR workSpace,
                                                  GM_ADDR tilingGm)
{
    SetMaskNorm();
    SetAtomicNone();

    this->coreIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    this->inAGM.SetGlobalBuffer((__gm__ T*)dA);
    this->uniRealGM.SetGlobalBuffer((__gm__ float*)dUniReal);
    this->outGM.SetGlobalBuffer((__gm__ T*)dAinv);
    this->workGM.SetGlobalBuffer((__gm__ float*)workSpace);

    InitUbuf();
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::InitUbuf()
{
    /*
        one repeat processes BASE_FP32_NUM (8192) fp32 elements:
        can be 8 * 32 * 32 ~ 128 * 8 * 8 (matrix num x n x n)
        the batch size per repeat ranges [8, 128]
        the size of matrix ranges [2, 32] (host guarantees n < 32 for this kernel)
    */
    this->alignedMatSize = (this->n + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;

    this->batchNumPerRepeat = BASE_FP32_NUM / (this->n * this->alignedMatSize); // [8, 128]
    if (this->batchNumPerRepeat > MAX_BATCH_NUM) {
        this->batchNumPerRepeat = MAX_BATCH_NUM;
    }
    this->batchNumPerRepeat = this->batchNumPerRepeat / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;

    // offset of one matrix in ub: n x alignedMatSize (fp32 elements)
    this->oneMatUbOffsetFp32 = this->n * this->alignedMatSize;

    this->pipe.InitBuffer(this->uBuf, MAX_UBUF_SIZE);
    LocalTensor<float> ubufLocal = uBuf.Get<float>();

    // init main buf: 3 planes x 32KB
    uint32_t allocOffset = 0;
    this->matARealLocal = ubufLocal[allocOffset];
    allocOffset += BASE_FP32_NUM;
    this->invRealLocal = ubufLocal[allocOffset];
    allocOffset += BASE_FP32_NUM;
    this->matRealLocal = ubufLocal[allocOffset];
    allocOffset += BASE_FP32_NUM;

    // init temp buf
    InitTempUbuf(ubufLocal, allocOffset);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::InitTempUbuf(LocalTensor<float> ubufLocal, uint32_t allocOffset)
{
    uint32_t alignedBatchSize = (this->batchNumPerRepeat + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32 *
                                ELENUM_REPEAT_FP32; // 256B aligned

    this->collectScalarOffsetLocal = ubufLocal[allocOffset].template ReinterpretCast<uint32_t>();
    allocOffset += alignedBatchSize; // 256 bytes

    this->preLocal = ubufLocal[allocOffset]; // pivot per batch
    allocOffset += alignedBatchSize;
    this->afterLocal = ubufLocal[allocOffset]; // a_ji per batch
    allocOffset += alignedBatchSize;
    this->resLocal = ubufLocal[allocOffset]; // factor per batch
    allocOffset += alignedBatchSize;

    uint32_t brcbEleNum = alignedBatchSize * ELENUM_BLOCK_FP32; // broadcast per row
    this->preBrcbLocal = ubufLocal[allocOffset];
    allocOffset += brcbEleNum;
    this->scalarBrcbLocal = ubufLocal[allocOffset];
    allocOffset += brcbEleNum;

    uint32_t updateTempEleNum = this->alignedMatSize * this->batchNumPerRepeat;
    this->updateTempLocal = ubufLocal[allocOffset];
    allocOffset += updateTempEleNum;
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t*>(tilingGm);
    this->isFp32Input = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingBuf)) == 0; // 0-fp32; 1-fp16
    uint32_t locId = 1;
    this->n = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingBuf + locId * sizeof(uint32_t)));
    locId++;
    this->offset = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingBuf + locId * sizeof(uint32_t) +
                                        sizeof(uint32_t) * this->coreIdx));
    locId += MAX_CORE_CNT;
    this->calNum = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingBuf + locId * sizeof(uint32_t) +
                                        sizeof(uint32_t) * this->coreIdx));
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::Process()
{
    auto eventId = EVENT_ID0;
    uint32_t overallRepeatTime = this->calNum / this->batchNumPerRepeat;
    uint32_t remainBatchNum = this->calNum % this->batchNumPerRepeat;
    if (remainBatchNum > 0) {
        overallRepeatTime++;
    }

    uint32_t batchOffset = this->batchNumPerRepeat * this->n * this->n;
    uint32_t currentBatchSize = this->batchNumPerRepeat;
    uint32_t currOffset = this->offset;
    for (uint32_t i = 0; i < overallRepeatTime; i++) {
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        if (remainBatchNum > 0 && i == overallRepeatTime - 1) {
            currentBatchSize = remainBatchNum;
        }

        // copy A batch in
        if constexpr (IS_FLOAT) {
            CopyMatBatchGmToUb(this->matARealLocal, this->inAGM[currOffset], this->n, currentBatchSize);
        } else {
            LocalTensor<T> inLocal = this->matRealLocal.template ReinterpretCast<T>();
            CopyMatBatchGmToUb(inLocal, this->inAGM[currOffset], this->n, currentBatchSize);

            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            PrepareDataForHalf(this->n, currentBatchSize);
        }

        // identity matrices for inverse accumulator
        CopyUniMatBatchGmToUb(this->invRealLocal, this->uniRealGM, this->alignedMatSize, currentBatchSize);

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        SingleProcess();

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        // copy out
        Copyout(currOffset, currentBatchSize);
        currOffset += batchOffset;
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::SingleProcess()
{
    /*
        Gauss-Jordan elimination in-place on matARealLocal (both fp32 and half paths:
        half input has already been cast into matARealLocal by PrepareDataForHalf).
        matRealLocal is unused here for fp32 (kept for UB layout symmetry with the
        complex kernel; the third plane stays free for future row-pivot support).
    */
    ComputeScalarOffset();
    PipeBarrier<PIPE_ALL>();

    for (uint16_t elementIdx = 0; elementIdx < this->n; elementIdx++) {
        // iterate diagonal elements
        // collect pivot a_ii of each matrix, then broadcast it per row
        uint32_t preOffset = elementIdx * this->alignedMatSize + elementIdx;
        CollectScalarSingle(this->preLocal, preOffset);
        PipeBarrier<PIPE_V>();
        BrcbScalar(this->preBrcbLocal, this->preLocal);

        uint32_t eleOffset = elementIdx * this->alignedMatSize;
        LocalTensor<float> tmpAi = this->matARealLocal[eleOffset];
        LocalTensor<float> tmpAiInv = this->invRealLocal[eleOffset];

        for (uint16_t rowIdx = 0; rowIdx < this->n; rowIdx++) {
            // iterate rows below/above pivot in i-th column
            if (rowIdx == elementIdx) {
                continue;
            }

            uint32_t rowOffset = rowIdx * this->alignedMatSize;
            LocalTensor<float> tmpAj = this->matARealLocal[rowOffset];
            LocalTensor<float> tmpAjInv = this->invRealLocal[rowOffset];

            // collect a_ji
            uint32_t afterOffset = rowIdx * this->alignedMatSize + elementIdx;
            CollectScalarSingle(this->afterLocal, afterOffset);
            PipeBarrier<PIPE_V>();

            // factor = a_ji / a_ii
            ScalarDiv();
            PipeBarrier<PIPE_V>();

            // A[j,:] = A[j,:] - factor * A[i,:]
            RealScalarVecBatchMul(this->updateTempLocal, tmpAi, this->scalarBrcbLocal, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();
            ComputeAndUpdateAj(tmpAj, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();

            // inv[j,:] = inv[j,:] - factor * inv[i,:]
            RealScalarVecBatchMul(this->updateTempLocal, tmpAiInv, this->scalarBrcbLocal, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();
            ComputeAndUpdateAj(tmpAjInv, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();
        }

        // A[i,:] = A[i,:] / a_ii
        PlaneRowVecScalarBatchDiv(tmpAi, this->preBrcbLocal, this->oneMatUbOffsetFp32);
        PipeBarrier<PIPE_V>();

        // inv[i,:] = inv[i,:] / a_ii
        PlaneRowVecScalarBatchDiv(tmpAiInv, this->preBrcbLocal, this->oneMatUbOffsetFp32);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::CopyMatBatchGmToUb(LocalTensor<T> dst, GlobalTensor<T> src,
                                                                uint32_t matColSize, uint32_t batchSize)
{
    uint32_t matRowSize = this->n;
    uint16_t blockCnt = matRowSize * batchSize;
    uint32_t blockLen = matColSize * sizeof(T);
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    if constexpr (IS_FLOAT) {
        // fp32 row = 4n bytes; hardware pads each row to ceil(4n/32)*32 = alignedMatSize*4 bytes,
        // which equals the target UB row stride -> dstStride is always 0.
        dstStride = 0;
    } else {
        // half row = 2n bytes; make each row land on alignedMatSize*4-byte boundary:
        // dstStride(blocks) = alignedMatSize/8 - ceil(2n/32)
        uint32_t srcRowBlocks = (blockLen + BYTENUM_BLOCK - 1) / BYTENUM_BLOCK;
        dstStride = this->alignedMatSize / ELENUM_BLOCK_FP32 - srcRowBlocks;
    }

    DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, dstStride, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::CopyUniMatBatchGmToUb(LocalTensor<float> dst, GlobalTensor<float> src,
                                                                   uint32_t matColSize, uint32_t batchSize)
{
    /*
        copy pre-built identity matrices (aligned rows) from plan GM into UB
    */
    uint32_t matRowSize = this->n;
    uint16_t blockCnt = matRowSize * batchSize;
    uint32_t blockLen = matColSize * sizeof(float);
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;

    DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, dstStride, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::CopyVecGmToUbUint32(LocalTensor<uint32_t> dst, GlobalTensor<uint32_t> src,
                                                                 uint32_t len)
{
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(uint32_t);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPadExtParams<uint32_t> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::CopyVecGmToUb(LocalTensor<float> dst, GlobalTensor<float> src,
                                                           uint32_t len)
{
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(float);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::CopyVecUbToGm(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t len)
{
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(T);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPad(dst, src, copyParams);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::ComputeScalarOffset()
{
    LocalTensor<uint32_t> offsetLocal = this->collectScalarOffsetLocal;
    uint32_t oneMatOffset = this->oneMatUbOffsetFp32;
    uint32_t batchSize = this->batchNumPerRepeat;

    uint32_t alignedBatchSize = (batchSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32 * ELENUM_REPEAT_FP32;

    for (uint32_t j = 0; j < alignedBatchSize; j++) {
        uint32_t offsetVal = 0;
        if (j < batchSize) {
            offsetVal = oneMatOffset * j * sizeof(float);
        } else {
            offsetVal = 0; // useless num, set to 0
        }
        offsetLocal.SetValue(j, offsetVal);
    }
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::PrepareDataForHalf(uint32_t matColSize, uint32_t batchSize)
{
    /*
        cast half input into fp32 buffer matARealLocal, row by row.
        the half rows were staged in matRealLocal (T view) by CopyMatBatchGmToUb
        at 32B-block granularity: row (m, r) starts at byte offset
        (m * n + r) * alignedMatSize * 4 and holds srcRowBlocks blocks of data,
        mirroring the fp32 row base (m * oneMatOffset + r * alignedMatSize).
        each row casts its staged blocks (srcRowBlocks * 16 half elements);
        lanes beyond n keep staging garbage, which either stays inside padding
        columns (column-isolated, harmless) or is overwritten by the next
        row's cast. a flat whole-plane cast cannot work here: half rows are
        16 elements per 32B block while fp32 rows are 8, so the element
        mapping would be wrong for every n.
    */
    LocalTensor<half> src = this->matRealLocal.template ReinterpretCast<half>();
    LocalTensor<float> dst = this->matARealLocal;

    uint32_t matRowSize = this->n;
    uint32_t rowCastCount = this->alignedMatSize; // fp32 elements per row (>= n, 8-aligned)

    for (uint32_t m = 0; m < batchSize; m++) {
        for (uint32_t r = 0; r < matRowSize; r++) {
            uint32_t srcOffset = (m * matRowSize + r) * 2 * this->alignedMatSize;
            uint32_t dstOffset = m * this->oneMatUbOffsetFp32 + r * this->alignedMatSize;
            Cast(dst[dstOffset], src[srcOffset], RoundMode::CAST_NONE, rowCastCount);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::CollectScalarSingle(LocalTensor<float> collectDst, uint32_t sOffset)
{
    /*
        collect one scalar from each matrix in UB (matARealLocal)
    */
    LocalTensor<float> matInner = this->matARealLocal;
    LocalTensor<uint32_t> offsetInner = this->collectScalarOffsetLocal;
    uint32_t batchSize = this->batchNumPerRepeat;

    uint8_t repeatTime = (batchSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;
    int32_t calCount = repeatTime * ELENUM_REPEAT_FP32;

    Gather(collectDst, matInner, offsetInner, sOffset * sizeof(float), calCount);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::BrcbScalar(LocalTensor<float> brcbDst, LocalTensor<float> scalarSrc)
{
    /*
        broadcast per-batch scalars to per-row layout: [s1 x 8, s2 x 8, ...]
    */
    uint32_t len = this->batchNumPerRepeat;
    uint8_t brcbRepeatTime = (len + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;
    Brcb(brcbDst, scalarSrc, brcbRepeatTime, {1, 8});
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::ScalarDiv()
{
    /*
        factor = a_ji / a_ii (per batch element)
    */
    uint32_t len = this->batchNumPerRepeat;
    uint32_t repeatTime = (len + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;

    Div(this->resLocal, this->afterLocal, this->preLocal, ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    // broadcast factor to per-row layout: [f1 x 8, f2 x 8, ...]
    BrcbScalar(this->scalarBrcbLocal, this->resLocal);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::RealScalarVecBatchMul(LocalTensor<float> dst, LocalTensor<float> src0,
                                                                   LocalTensor<float> src1, uint32_t oneMatOffset)
{
    uint32_t matAlignedSize = this->alignedMatSize;
    uint32_t batchSize = this->batchNumPerRepeat;
    uint8_t oneRowBlockStride = matAlignedSize / ELENUM_BLOCK_FP32;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32;
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Mul(dst[ELENUM_BLOCK_FP32 * matAlignedSize * j], src0[ELENUM_BLOCK_FP32 * oneMatOffset * j],
                src1[ELENUM_REPEAT_FP32 * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneRowBlockStride, oneMatBlockStride, 1, 1, 1, 0});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Mul(dst[ELENUM_BLOCK_FP32 * i], src0[ELENUM_BLOCK_FP32 * i], src1, ELENUM_REPEAT_FP32,
                (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneRowBlockStride, oneMatBlockStride, 1, static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneRowBlockStride),
                 static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneMatBlockStride), 8});
        }
    }
    PipeBarrier<PIPE_V>();
}

// dst = dst - src
template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::RealScalarVecBatchSub(LocalTensor<float> dst, LocalTensor<float> src,
                                                                   uint32_t oneMatOffset)
{
    uint32_t matAlignedSize = this->alignedMatSize;
    uint32_t batchSize = this->batchNumPerRepeat;
    uint8_t oneRowBlockStride = matAlignedSize / ELENUM_BLOCK_FP32;
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32;
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Sub(dst[ELENUM_BLOCK_FP32 * oneMatOffset * j], dst[ELENUM_BLOCK_FP32 * oneMatOffset * j],
                src[ELENUM_BLOCK_FP32 * matAlignedSize * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneMatBlockStride, oneMatBlockStride, oneRowBlockStride, 1, 1, 1});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Sub(dst[ELENUM_BLOCK_FP32 * i], dst[ELENUM_BLOCK_FP32 * i], src[ELENUM_BLOCK_FP32 * i], ELENUM_REPEAT_FP32,
                (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneMatBlockStride, oneMatBlockStride, oneRowBlockStride,
                 static_cast<uint8_t>(oneMatBlockStride * ELENUM_BLOCK_FP32),
                 static_cast<uint8_t>(oneMatBlockStride * ELENUM_BLOCK_FP32),
                 static_cast<uint8_t>(oneRowBlockStride * ELENUM_BLOCK_FP32)});
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::ComputeAndUpdateAj(LocalTensor<float> Aj, uint32_t oneMatOffset)
{
    /*
        A[j,:] = A[j,:] - factor * A[i,:]
        (updateTempLocal already holds factor * A[i,:])
    */
    RealScalarVecBatchSub(Aj, this->updateTempLocal, oneMatOffset);
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::PlaneRowVecScalarBatchDiv(LocalTensor<float> dst, LocalTensor<float> src1,
                                                                       uint32_t oneMatOffset)
{
    /*
        plane row chunk update: A[i,:] = A[i,:] / a_ii (in place)
        dst addresses row i of matrix 0; the strided access reaches row i of
        every matrix in the batch. src1 holds the broadcast pivots
        [p1 x 8, p2 x 8, ...] (one 32B block per matrix), so dst and src0
        share the matrix-major block strides while src1 advances one block
        per matrix.
    */
    uint32_t matAlignedSize = this->alignedMatSize;
    uint32_t batchSize = this->batchNumPerRepeat;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32;
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Div(dst[ELENUM_BLOCK_FP32 * oneMatOffset * j], dst[ELENUM_BLOCK_FP32 * oneMatOffset * j],
                src1[ELENUM_REPEAT_FP32 * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneMatBlockStride, oneMatBlockStride, 1, 1, 1, 0});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Div(dst[ELENUM_BLOCK_FP32 * i], dst[ELENUM_BLOCK_FP32 * i], src1, ELENUM_REPEAT_FP32,
                (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneMatBlockStride, oneMatBlockStride, 1, static_cast<uint8_t>(oneMatBlockStride * ELENUM_BLOCK_FP32),
                 static_cast<uint8_t>(oneMatBlockStride * ELENUM_BLOCK_FP32), 8});
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SmatinvBatchedAIV<T>::Copyout(uint32_t outOffset, uint32_t batchSize)
{
    GlobalTensor<T> gmMatOutInner = this->outGM[outOffset];
    LocalTensor<float> invInner = this->invRealLocal;
    auto eventId = EVENT_ID0;

    uint32_t matRowSize = this->n;
    uint32_t matColSize = this->n;
    uint32_t matColFpAlignedSize = this->alignedMatSize;

    if constexpr (IS_FLOAT) {
        // row-by-row copy back: aligned rows in UB -> packed rows in GM
        // srcStride(blocks) = alignedRow/8 - ceil(4n/32); GM rows are packed (dstStride=0)
        uint16_t blockCnt = matRowSize * batchSize;
        uint32_t blockLen = matColSize * sizeof(float);
        uint32_t rowBlocks = (blockLen + BYTENUM_BLOCK - 1) / BYTENUM_BLOCK;
        uint32_t srcStride = matColFpAlignedSize / ELENUM_BLOCK_FP32 - rowBlocks;

        DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, 0, 0};
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        DataCopyPad(gmMatOutInner, invInner, copyParams);
    } else {
        /*
            cast fp32 -> half row-wise into the free plane (matRealLocal, T-view),
            then pack out with one strided DataCopyPad.
            staging rows are 32B-block aligned: row (m, r) occupies rowBlocks 32B
            blocks starting at half offset (m * n + r) * rowBlocks * 16 (a packed
            m*n-per-row layout would misalign row bases for n % 16 != 0).
            cast writes alignedMatSize half per row (<= rowBlocks * 16, proven:
            ceil(2n/32)*16 >= alignedMatSize for all supported n), so the cast
            base addresses stay 32B aligned and padding lanes stay inside the row.
        */
        LocalTensor<half> castDstLocal = this->matRealLocal.template ReinterpretCast<half>();
        uint32_t rowBlocks = (matColFpAlignedSize * sizeof(half) + BYTENUM_BLOCK - 1) / BYTENUM_BLOCK;
        for (uint32_t m = 0; m < batchSize; m++) {
            uint32_t matBaseFp = m * this->oneMatUbOffsetFp32;
            uint32_t stageBaseH = m * matRowSize * rowBlocks * ELENUM_BLOCK_FP16;
            for (uint32_t r = 0; r < matRowSize; r++) {
                Cast(castDstLocal[stageBaseH + r * rowBlocks * ELENUM_BLOCK_FP16],
                     invInner[matBaseFp + r * matColFpAlignedSize], RoundMode::CAST_NONE, matColFpAlignedSize);
            }
        }
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        // pack out: one pass over staging rows; skip in-row tail blocks
        uint16_t blockCnt = matRowSize * batchSize;
        uint32_t blockLen = matColSize * sizeof(T);
        uint32_t srcStride = rowBlocks - (blockLen + BYTENUM_BLOCK - 1) / BYTENUM_BLOCK;
        DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, 0, 0};
        DataCopyPad(gmMatOutInner, castDstLocal, copyParams);
    }
}

} // namespace SmatinvBatched

#endif // SMATINV_BATCHED_AIV_H
