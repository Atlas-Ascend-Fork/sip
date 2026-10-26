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

/*
 * ===================== real-input variants (SgetriBatched) =====================
 * 实数输入布局: A_org 为 [B, n, n] fp32 连续 (无交错)。
 * A_work 布局与复数版一致 ([real M x strideN | imag M x strideN])。
 * pad_real: real 直拷入 real plane (无 GatherMask 拆分), imag plane 写零 ——
 *   下游 lu/trsm/getri 全部照旧 (imag 恒 0 精确传播, v1 c64-wrap 已端到端验证)。
 * rebuild_eye_real: 每次执行前重建 eye_gm 为单位阵布局
 *   ([real M x strideN | imag M x strideN | imagNeg M x strideN], real 段对角
 *   置 1 其余 0, imag/imagNeg 段全 0)。修复 plan 复用脏数据: eye_gm 被 kernel
 *   原位消耗 (SwapRows 行交换 + trsm 原位求解), host 只在 MakePlan 上载一次,
 *   复数版同样受影响 (上游遗留问题)。
 * restore_real: 只拷回 real plane (结果在 eye_gm 的 real 段, 与复数 restore 一致)。
 *
 * 搬运方式: GM<->GM 无直连通路, 经 UB 中转 (VECIN MTE2 搬入 -> V 管道 ->
 * VECOUT MTE3 搬出)。GM 侧行距 (strideN) 与密排 (orgN) 均通过张量下标逐行
 * 寻址, DataCopyExtParams 固定 {1, len, 0, 0, 0} 最简形式 (与 cast.hpp 的
 * cast_up/cast_down 完全同款, 该模式在 c32 路径已长期验证), 规避 srcStride/
 * dstStride 字节-vs-块 单位歧义。逐行 MTE 指令开销相对 LU 主体可忽略。
 * 与复数版一致, GM 只写 [0, orgM) x [0, orgN) 有效区 (padding 区靠 A_work
 * 零初始化 + 下游不写保持)。
 */

class PadReal {
    static constexpr int TILE_ELE = 8192;
    GlobalTensor<float> srcGm, dstRealGm, dstImagGm;
    int orgN;
    int strideN;
    int linesPerIter;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;

public:
    __aicore__ inline PadReal() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<float> src,
                                GlobalTensor<float> dstReal, GlobalTensor<float> dstImag, int orgM, int orgN,
                                int strideN)
    {
        this->srcGm = src;
        this->dstRealGm = dstReal;
        this->dstImagGm = dstImag;
        this->orgN = orgN;
        this->strideN = strideN;
        this->linesPerIter = max(1, TILE_ELE / strideN);
        pipe->InitBuffer(inQueue, 2, linesPerIter * strideN * sizeof(float));
        pipe->InitBuffer(outQueue, 2, linesPerIter * strideN * sizeof(float));
    }
    __aicore__ inline void RealWork(int srcRow, int dstRow, int lines)
    {
        int done = 0;
        while (done < lines) {
            int cur = min(linesPerIter, lines - done);
            CopyInRows(srcRow + done, cur);
            Relay(cur);
            CopyOutRealRows(dstRow + done, cur);
            // A4 终版修复: 每迭代全屏障 —— 迭代内 MTE2/V/MTE3 流水照常,
            // 迭代结束全管排空后再回绕, 双缓冲复用与在飞读写绝对无窗口。
            // (代价: pad 段丢失跨迭代流水; eye/pad/restore 为初始化/收尾
            //  段, 占整体耗时 <5%, 且实测事件对逐边补法会扰动时序开新窗,
            //  粗粒度全屏障是唯一确定闭合的方案)
            PipeBarrier<PIPE_ALL>();
            done += cur;
        }
    }
    __aicore__ inline void RealWorkSplit(int M, int p, int id)
    {
        int k = M / p;
        int r = M % p;
        RealWork(k * id + min(id, r), k * id + min(id, r), k + (id < r));
    }
    __aicore__ inline void ImagZeroWork(int dstRow, int lines)
    {
        int done = 0;
        while (done < lines) {
            int cur = min(linesPerIter, lines - done);
            Zero();
            CopyOutImagRows(dstRow + done, cur);
            PipeBarrier<PIPE_ALL>(); // A4: 每迭代全屏障 (同 RealWork)
            done += cur;
        }
    }
    __aicore__ inline void ImagZeroWorkSplit(int M, int p, int id)
    {
        int k = M / p;
        int r = M % p;
        ImagZeroWork(k * id + min(id, r), k + (id < r));
    }
    __aicore__ inline void CopyInRows(int srcRow, int lines)
    {
        LocalTensor<float> srcLocal = inQueue.AllocTensor<float>();
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(orgN * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        for (int r = 0; r < lines; ++r) {
            // GM 密排 [orgM, orgN] -> UB staging 行距 strideN (逐行寻址)
            DataCopyPad(srcLocal[r * strideN], srcGm[(srcRow + r) * orgN], rowParams, padParams);
        }
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Relay(int lines)
    {
        LocalTensor<float> srcLocal = inQueue.DeQue<float>();
        LocalTensor<float> relayLocal = outQueue.AllocTensor<float>();
        DataCopy(relayLocal, srcLocal, lines * strideN);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(relayLocal);
    }
    __aicore__ inline void CopyOutRealRows(int dstRow, int lines)
    {
        LocalTensor<float> relayLocal = outQueue.DeQue<float>();
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(orgN * sizeof(float)), 0, 0, 0};
        for (int r = 0; r < lines; ++r) {
            DataCopyPad(dstRealGm[(dstRow + r) * strideN], relayLocal[r * strideN], rowParams);
        }
        outQueue.FreeTensor(relayLocal);
    }
    __aicore__ inline void Zero()
    {
        LocalTensor<float> zeroLocal = outQueue.AllocTensor<float>();
        Duplicate(zeroLocal, (float)0.0, linesPerIter * strideN);
        PipeBarrier<PIPE_V>();
        outQueue.EnQue(zeroLocal);
    }
    __aicore__ inline void CopyOutImagRows(int dstRow, int lines)
    {
        LocalTensor<float> zeroLocal = outQueue.DeQue<float>();
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(orgN * sizeof(float)), 0, 0, 0};
        for (int r = 0; r < lines; ++r) {
            DataCopyPad(dstImagGm[(dstRow + r) * strideN], zeroLocal[r * strideN], rowParams);
        }
        outQueue.FreeTensor(zeroLocal);
    }
};

__aicore__ inline void pad_real(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, GM_ADDR A_org,
                                GM_ADDR A_work)
{
#ifdef __DAV_C220_VEC__
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobalReal;
    GlobalTensor<float> aGlobalImag;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_org));
    aGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_work));
    aGlobalImag = aGlobalReal[strideN * M];

    PadReal op;
    op.Init(&tbufPool, aGlobalOrg, aGlobalReal, aGlobalImag, orgM, orgN, strideN);
    op.RealWorkSplit(orgM, GetBlockNum() << 1, GetBlockIdx());
    op.ImagZeroWorkSplit(orgM, GetBlockNum() << 1, GetBlockIdx());

    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
#endif
}

class RestoreReal {
    static constexpr int TILE_ELE = 8192;
    GlobalTensor<float> srcGm, dstGm;
    int orgN;
    int strideN;
    int linesPerIter;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;

public:
    __aicore__ inline RestoreReal() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<float> src,
                                GlobalTensor<float> dst, int orgM, int orgN, int strideN)
    {
        this->srcGm = src;
        this->dstGm = dst;
        this->orgN = orgN;
        this->strideN = strideN;
        this->linesPerIter = max(1, TILE_ELE / strideN);
        pipe->InitBuffer(inQueue, 2, linesPerIter * strideN * sizeof(float));
        pipe->InitBuffer(outQueue, 2, linesPerIter * strideN * sizeof(float));
    }
    __aicore__ inline void Process(int srcRow, int dstRow, int lines)
    {
        int done = 0;
        while (done < lines) {
            int cur = min(linesPerIter, lines - done);
            CopyInRows(srcRow + done, cur);
            Relay(cur);
            CopyOutRows(dstRow + done, cur);
            PipeBarrier<PIPE_ALL>(); // A4: 每迭代全屏障 (同 PadReal::RealWork)
            done += cur;
        }
    }
    __aicore__ inline void ProcessSplit(int M, int p, int id)
    {
        int k = M / p;
        int r = M % p;
        Process(k * id + min(id, r), k * id + min(id, r), k + (id < r));
    }
    __aicore__ inline void CopyInRows(int srcRow, int lines)
    {
        LocalTensor<float> srcLocal = inQueue.AllocTensor<float>();
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(orgN * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        for (int r = 0; r < lines; ++r) {
            // eye real plane 行距 strideN -> UB staging 行距 strideN
            DataCopyPad(srcLocal[r * strideN], srcGm[(srcRow + r) * strideN], rowParams, padParams);
        }
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Relay(int lines)
    {
        LocalTensor<float> srcLocal = inQueue.DeQue<float>();
        LocalTensor<float> relayLocal = outQueue.AllocTensor<float>();
        DataCopy(relayLocal, srcLocal, lines * strideN);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(relayLocal);
    }
    __aicore__ inline void CopyOutRows(int dstRow, int lines)
    {
        LocalTensor<float> relayLocal = outQueue.DeQue<float>();
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(orgN * sizeof(float)), 0, 0, 0};
        for (int r = 0; r < lines; ++r) {
            // UB staging -> A_inv 密排 [orgM, orgN] (逐行寻址)
            DataCopyPad(dstGm[(dstRow + r) * orgN], relayLocal[r * strideN], rowParams);
        }
        outQueue.FreeTensor(relayLocal);
    }
};

__aicore__ inline void restore_real(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, GM_ADDR A_inv,
                                    GM_ADDR eye_gm)
{
#ifdef __DAV_C220_VEC__
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);

    int strideN = (orgN + 127) / 128 * 128;

    GlobalTensor<float> aGlobalInv;
    GlobalTensor<float> eyeGlobalReal;
    aGlobalInv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_inv));
    eyeGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eye_gm));

    RestoreReal op;
    op.Init(&tbufPool, eyeGlobalReal, aGlobalInv, orgM, orgN, strideN);
    op.ProcessSplit(orgM, GetBlockNum() << 1, GetBlockIdx());

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
            // A4: 每迭代全屏障 (同 PadReal/RestoreReal) —— MTE3 写 eye 排空
            // 后才回绕复用, eye 对角不被在飞读写破坏 (1/n 型误差源头之一)
            PipeBarrier<PIPE_ALL>();
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

#endif // _PAD_HPP_
