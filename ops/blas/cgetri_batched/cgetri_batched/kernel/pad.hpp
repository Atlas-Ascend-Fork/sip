/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _PAD_HPP_
#define _PAD_HPP_

#include <cstdint>
#include <lib/matrix/matmul/matmul.h>
#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

template <typename T>
class SplitRealImag {
    static const int TILE_LENGTH = 8192;
    GlobalTensor<T> srcGlobal, dstGlobalReal, dstGlobalImag;
    int N;
    int srcN;
    int dstN;
    int padN;
    int alignedN;
    int linesPerIter;
    int elementsPerBlock;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueueReal, outQueueImag;
    LocalTensor<T> realLocal, imagLocal;

public:
    __aicore__ inline SplitRealImag() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe)
    {
        pipe->InitBuffer(inQueue, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueReal, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueImag, 2, TILE_LENGTH * sizeof(T));
        elementsPerBlock = 32 / sizeof(T);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobalReal,
                                     GlobalTensor<T> dstGlobalImag, int N, int srcN, int dstN)
    {
        this->srcGlobal = srcGlobal;
        this->dstGlobalReal = dstGlobalReal;
        this->dstGlobalImag = dstGlobalImag;
        this->N = N;
        this->srcN = srcN;
        this->dstN = dstN;
        alignedN = (N + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        linesPerIter = TILE_LENGTH / alignedN;
        linesPerIter = max(1, linesPerIter & ~1);
        padN = (elementsPerBlock - (N * 2 % elementsPerBlock)) % elementsPerBlock;
    }
    __aicore__ inline void SplitRealImagWork(int M, int p, int id)
    {
        int k = M / p;
        int r = M % p;
        Process((k * id + min(id, r)) * srcN * 2, (k * id + min(id, r)) * dstN, k + (id < r));
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines)
    {
        int loopCount = lines / linesPerIter;
        srcOffset += loopCount * linesPerIter * srcN * 2;
        dstOffset += loopCount * linesPerIter * dstN;
        int tailLines = lines - linesPerIter * loopCount;
        if (tailLines) {
            if (N << 1 <= TILE_LENGTH) {
                CopyIn(srcOffset, tailLines / 2);
                Compute(0, 0, tailLines / 2 * alignedN * 2);
                CopyIn(srcOffset + tailLines / 2 * srcN * 2, (tailLines + 1) / 2);
                Compute(1, tailLines / 2 * alignedN, (tailLines + 1) / 2 * alignedN * 2);
            } else {
                CopyIn(srcOffset, TILE_LENGTH);
                Compute(0, 0, TILE_LENGTH);
                CopyIn(srcOffset + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
                Compute(1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
            }
            CopyOut(dstOffset, tailLines);
        }
        srcOffset -= linesPerIter * srcN * 2;
        dstOffset -= linesPerIter * dstN;
        for (int i = 0; i < loopCount; ++i) {
            if (N << 1 <= TILE_LENGTH) {
                CopyIn(srcOffset, linesPerIter / 2);
                Compute(0, 0, linesPerIter / 2 * alignedN * 2);
                CopyIn(srcOffset + linesPerIter / 2 * srcN * 2, (linesPerIter + 1) / 2);
                Compute(1, linesPerIter / 2 * alignedN, (linesPerIter + 1) / 2 * alignedN * 2);
            } else {
                CopyIn(srcOffset, TILE_LENGTH);
                Compute(0, 0, TILE_LENGTH);
                CopyIn(srcOffset + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
                Compute(1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
            }
            CopyOut(dstOffset, linesPerIter);
            srcOffset -= linesPerIter * srcN * 2;
            dstOffset -= linesPerIter * dstN;
        }
    }
    __aicore__ inline void CopyIn(int offset, int lines)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        if (N << 1 <= TILE_LENGTH) {
            DataCopyExtParams intriParams{
                static_cast<uint16_t>(lines), static_cast<uint16_t>(N * 2 * sizeof(T)),
                static_cast<uint16_t>((srcN * 2 - N * 2) * sizeof(T)),
                static_cast<uint16_t>(N % elementsPerBlock && N % elementsPerBlock <= elementsPerBlock / 2), 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padN), 0};
            DataCopyPad(srcLocal, srcGlobal[offset], intriParams, padParams);
        } else {
            DataCopy(srcLocal, srcGlobal[offset], lines);
        }
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int progress, int calcedLength, int length)
    {
        if (!progress) {
            realLocal = outQueueReal.AllocTensor<T>();
            imagLocal = outQueueImag.AllocTensor<T>();
        }
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        uint64_t rsvdCnt;
        GatherMask(realLocal[calcedLength], srcLocal, 1, false, 0,
                   {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        GatherMask(imagLocal[calcedLength], srcLocal, 2, false, 0,
                   {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(srcLocal);
        if (progress) {
            outQueueReal.EnQue(realLocal);
            outQueueImag.EnQue(imagLocal);
        }
    }
    __aicore__ inline void CopyOut(int offset, int lines)
    {
        LocalTensor<T> realLocal = outQueueReal.DeQue<T>();
        LocalTensor<T> imagLocal = outQueueImag.DeQue<T>();
        DataCopyExtParams intriParams{static_cast<uint16_t>(lines), static_cast<uint16_t>(N * sizeof(T)), 0,
                                      static_cast<uint16_t>((dstN - N) * sizeof(T)), 0};
        DataCopyPad(dstGlobalReal[offset], realLocal, intriParams);
        DataCopyPad(dstGlobalImag[offset], imagLocal, intriParams);
        outQueueReal.FreeTensor(realLocal);
        outQueueImag.FreeTensor(imagLocal);
    }
};

template <typename T>
class MergeRealImag {
    static const int TILE_LENGTH = 8192;
    GlobalTensor<T> aOrgGlobal;
    GlobalTensor<T> aGlobalReal, aGlobalImag;
    GlobalTensor<uint32_t> aGather;
    int N;
    int alignedN;
    int srcN;
    int dstN;
    int linesPerIter;
    int elementsPerBlock;
    int baseN, aligned_n_cache, nFullBlocks, nTailBlocks;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TQue<QuePosition::VECIN, 1> gatherQueue;
    LocalTensor<T> realLocal, imagLocal;
    LocalTensor<uint32_t> gatherLocal;

public:
    __aicore__ inline MergeRealImag() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<uint32_t> aGather)
    {
        pipe->InitBuffer(inQueue, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueue, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(gatherQueue, 1, TILE_LENGTH * sizeof(uint32_t));
        elementsPerBlock = 32 / sizeof(T);
        this->aGather = aGather;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aOrgGlobal, GlobalTensor<T> aGlobalReal,
                                     GlobalTensor<T> aGlobalImag, int N, int srcN, int dstN)
    {
        this->aOrgGlobal = aOrgGlobal;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->N = N;
        this->alignedN = ceil(N, elementsPerBlock);
        this->srcN = srcN;
        this->dstN = dstN;
        linesPerIter = max(1, TILE_LENGTH / ceil(dstN, 16) / 2);
    }
    __aicore__ inline void MergeRealImagWork(int M, int p, int id)
    {
        int k = M / p;
        int r = M % p;
        Process((k * id + min(id, r)) * srcN, (k * id + min(id, r)) * dstN * 2, k + (id < r));
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines)
    {
        LoadGather();
        int loopCount = lines / linesPerIter;
        for (int i = 0; i < loopCount; ++i) {
            CopyIn(srcOffset, linesPerIter, alignedN);
            Compute(linesPerIter * alignedN * 2);
            CopyOut(dstOffset, linesPerIter, N);
            srcOffset += linesPerIter * srcN;
            dstOffset += linesPerIter * dstN * 2;
        }
        lines %= linesPerIter;
        if (!lines) {
            gatherQueue.FreeTensor(gatherLocal);
            return;
        }
        CopyIn(srcOffset, lines, alignedN);
        Compute(lines * alignedN * 2);
        CopyOut(dstOffset, lines, N);
        gatherQueue.FreeTensor(gatherLocal);
    }
    __aicore__ inline void LoadGather()
    {
        gatherLocal = gatherQueue.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, aGather, linesPerIter * alignedN * 2);
        gatherQueue.EnQue(gatherLocal);
        gatherLocal = gatherQueue.DeQue<uint32_t>();
    }
    __aicore__ inline void CopyIn(int offset, int blockM, int blockN)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();

        DataCopyParams copyInParams{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN / elementsPerBlock),
                                    static_cast<uint16_t>((srcN - blockN) / elementsPerBlock), 0};
        DataCopy(srcLocal, aGlobalReal[offset], copyInParams);
        DataCopy(srcLocal[TILE_LENGTH / 2], aGlobalImag[offset], copyInParams);

        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int length)
    {
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, length);
        PipeBarrier<PIPE_ALL>();
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut(int offset, int blockM, int blockN)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();

        DataCopyExtParams copyOutParams{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN * 2 * sizeof(T)),
                                        static_cast<uint16_t>((alignedN * 2 - blockN * 2) / elementsPerBlock),
                                        static_cast<uint16_t>((dstN * 2 - blockN * 2) * sizeof(T)), 0};
        DataCopyPad(aOrgGlobal[offset], dstLocal, copyOutParams);

        outQueue.FreeTensor(dstLocal);
    }
};

__aicore__ inline void pad(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, GM_ADDR A_org,
                           GM_ADDR A_work)
{
#ifdef __DAV_C220_VEC__
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobalReal;
    GlobalTensor<float> aGlobalImag;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_org));
    aGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_work));
    aGlobalImag = aGlobalReal[strideN * M];

    SplitRealImag<float> opsp;
    opsp.Init(&tbufPool);
    opsp.SetMatrix(aGlobalOrg, aGlobalReal, aGlobalImag, orgN, orgN, strideN);
    opsp.SplitRealImagWork(orgM, blockNum << 1, blockIdx);

    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
#endif
}

__aicore__ inline void restore(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, GM_ADDR A_org,
                               GM_ADDR A_work, GM_ADDR gather3_gm)
{
#ifdef __DAV_C220_VEC__
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);

    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobalReal;
    GlobalTensor<float> aGlobalImag;
    GlobalTensor<uint32_t> gather3;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_org));
    aGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_work));
    aGlobalImag = aGlobalReal[strideN * M];
    gather3.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gather3_gm));

    MergeRealImag<float> opmg;
    static constexpr int TILE_LENGTH = 4096;
    opmg.Init(&tbufPool, gather3);
    opmg.SetMatrix(aGlobalOrg, aGlobalReal, aGlobalImag, min(orgN, TILE_LENGTH), strideN, orgN);
    opmg.MergeRealImagWork(orgM, blockNum << 1, blockIdx);
    if (orgN > TILE_LENGTH) {
        opmg.SetMatrix(aGlobalOrg[TILE_LENGTH * 2], aGlobalReal[TILE_LENGTH], aGlobalImag[TILE_LENGTH],
                       orgN - TILE_LENGTH, strideN, orgN);
        opmg.MergeRealImagWork(orgM, blockNum << 1, blockIdx);
    }

    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
#endif
}

/*
 * eye_gm 重建 (每次 kernel 执行前调用, 修复 plan 复用脏数据)。
 * 几何: host 每批分配 eyeMatEleNum = strideN * (ceil16(n) + 128) * 3 个 float,
 *   kernel 侧以段步长 M*strideN 读 [real | imag | imagNeg] (M = ceil16(n)),
 *   每段尾部 128 行为 slack。host 只把 real 段 [0,n) 行对角线置 1,
 *   其余全 0 (上传时整个 buffer 零初始化)。
 * kernel 每次执行原位消耗 eye (SwapRows 行交换 + trsm 原位求解),
 * plan 复用 (BlasMixCache 按 opName+params 缓存 handle) 时第二次调用起
 * eye 已被污染 —— 必须每次整批全量清零 + 对角置 1 (等价 host 初态)。
 *
 * 对角置 1 必须与清零在同一 UB 块内完成 (Duplicate 后在 zeroLocal 上
 * SetValue, 再一并 MTE3 写出): GM SetValue 标量写与 VECOUT 队列的 MTE3
 * 异步写之间无顺序保证, 分离两步会竞态 (对角 1 被后落地的清零覆盖,
 * 实测 call#1 即错且错误值与 eye 脏数据一致)。
 */
class EyeReal {
    static constexpr int TILE_ELE = 8192;
    GlobalTensor<float> eyeGm;
    int orgN;
    int strideN;
    int linesPerIter;
    TQue<QuePosition::VECOUT, 2> outQueue;

public:
    __aicore__ inline EyeReal() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<float> eye, int orgN, int strideN)
    {
        this->eyeGm = eye;
        this->orgN = orgN;
        this->strideN = strideN;
        this->linesPerIter = max(1, TILE_ELE / strideN);
        pipe->InitBuffer(outQueue, 2, linesPerIter * strideN * sizeof(float));
    }
    __aicore__ inline void BuildRows(int startRow, int rows)
    {
        int done = 0;
        while (done < rows) {
            int cur = min(linesPerIter, rows - done);
            LocalTensor<float> blkLocal = outQueue.AllocTensor<float>();
            Duplicate(blkLocal, (float)0.0, linesPerIter * strideN);
            PipeBarrier<PIPE_V>();
            // real 段 [0, n) 行的对角元素: eye 行 g 的对角在列 g,
            // 本块行 g == startRow + done + r -> blkLocal[r*strideN + g] = 1
            // (Duplicate V 管道写与 SetValue 标量写之间必须 barrier, 否则
            //  标量写可被 Duplicate 覆盖 -> 单位阵对角丢失, 实测 call#1 即错)
            for (int r = 0; r < cur; ++r) {
                int g = startRow + done + r;
                if (g < orgN) {
                    blkLocal.SetValue(r * strideN + g, (float)1.0);
                }
            }
            PipeBarrier<PIPE_V>();
            outQueue.EnQue(blkLocal);
            blkLocal = outQueue.DeQue<float>();
            DataCopyExtParams rowParams{1, static_cast<uint32_t>(strideN * sizeof(float)), 0, 0, 0};
            for (int r = 0; r < cur; ++r) {
                DataCopyPad(eyeGm[(startRow + done + r) * strideN], blkLocal[r * strideN], rowParams);
            }
            outQueue.FreeTensor(blkLocal);
            done += cur;
        }
    }
    __aicore__ inline void BuildRowsSplit(int totalRows, int p, int id)
    {
        int k = totalRows / p;
        int r = totalRows % p;
        BuildRows(k * id + min(id, r), k + (id < r));
    }
};

__aicore__ inline void rebuild_eye_real(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, GM_ADDR eye_gm)
{
#ifdef __DAV_C220_VEC__
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;
    // host 每批分配总行数 (含三段各 128 行 slack), 全量重建最稳妥
    int totalRows = (M + 128) * 3;

    GlobalTensor<float> eyeGlobal;
    eyeGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eye_gm));

    EyeReal op;
    op.Init(&tbufPool, eyeGlobal, orgN, strideN);
    op.BuildRowsSplit(totalRows, GetBlockNum() << 1, GetBlockIdx());

    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
#endif
}

#endif // _PAD_HPP_
