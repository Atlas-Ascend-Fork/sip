/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GETRI_PER_CORE_HPP_
#define _GETRI_PER_CORE_HPP_

/*
 * P2 每核独立路径 (仅实数 fp32, dtype==2, n<=256):
 *
 * 背景: 主路径 LU 面板分解在 blockM<=512 分支只由 core0 执行
 * (lu_custom.hpp L120-124), 39/40 核全程空等; 批内矩阵串行。
 * P2: 每个 AIV 核独立认领矩阵序列 (bid = aivIdx, bid += aivNum),
 * 单核完成 eye->pad->LU(面板+行交换+U12+Schur)->下三角求解
 * ->上三角求解->restore 全流程, 零跨核同步。
 *
 * v1 上板 b=4 首调即 aicore timeout (core33 死等), 根因与修复:
 *  R1) [timeout 主因] pad.hpp 的 rebuild_eye_real/pad_real/restore_real
 *      内含 CrossCoreSetFlag/WaitFlag(0x2) 全核 barrier 且按
 *      GetBlockNum()<<1 分片 —— P2 下仅 aivIdx<batchSize 的核调用,
 *      其余核已 return, barrier 永远等不齐 -> 死等。v2 复用
 *      EyeReal/PadReal/RestoreReal 类的全量接口自建单核版 (无 barrier)。
 *  R2) 主路径 custom_lu 由 block8/block10 核按 W 应用 pivot 行交换
 *      (A 尾列段/L 列段/eye), v1 遗漏 -> 因子错乱。v2 逐面板单核补做,
 *      段范围与主路径逐项对齐 (A: [p+16,N2), L: [0,p), eye: [0,N2))。
 *  R3) 面板间 Schur 更新 (A22 -= L21*U12) 由主路径 CUBE matmul 承担,
 *      v1 完全没有 -> LU 因子错。v2: U12 = Lpanel^-1*A[面板行,尾列]
 *      由 PanelSolve 解出 (正号), Schur 用 TailUpdate 行块 Axpy。
 *  R4) v1 TailUpdate 面板 tensor 从 depth=1 队列 Alloc 后贯穿循环未
 *      Free, blk 再 Alloc 同队列 -> 死等。v2 面板改 TBuf 常驻; 行块/
 *      系数走独立队列, 显式事件同步 (LUCustom1 CopyOut 同款范式)。
 *  R5) v1 每面板 tbufPool.Reset() 吊销已 Init 的 buffer -> 越界。
 *      v2 阶段内 Init 一次循环只 Process; 阶段间 Reset 后重新 Init。
 *  R6) v1 SolveTrsmCustom 误传 M=ceil16(n) (应为行距 strideN)。
 *      v2 弃用之, PanelSolve 自带正确寻址。
 *  R7) v1 getf2 面板 srcOffset 漏乘行距 (应 offsetN*strideN+offsetN)。
 *  R8) 阶段/面板间 MTE3->MTE2、V->MTE3 跨管道事件补全 (主路径靠
 *      CrossCore flag 天然同步, P2 无 barrier 必须显式事件):
 *      pad->LU / Process1->LU / LU->swap / swap->U12 / U12->Schur /
 *      Schur->下一面板 / 面板解->尾端 / 尾端->下一面板 / tile->tile。
 *  R9) v1 主入口 aivNum=GetBlockNum(): VEC 侧 GetBlockNum()=blockDim
 *      (=20, pad.hpp 以 GetBlockNum()<<1 分片且主路径全量覆盖为证),
 *      而 GetBlockIdx()∈[0,40) -> 核 20~39 与 0~19 重复处理同一 bid,
 *      并发写同一 A_work 切片。v2 aivNum=GetBlockNum()*2; 切片步长
 *      补 +128 slack 与 host 一致 (v1 少 3*128*sN, 核间踩踏)。
 *  R10) 尾端更新系数改从 GM 逐 (i,p) 标量读 (~n^2 次/阶段, S 管 GM
 *      负载延迟会将 n=256 拖至毫秒级) 为每 tile 批量搬 16 列系数块
 *      入 UB (coeffQue) 后标量读。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "getf2.hpp"
#include "lu_custom.hpp" // pad.hpp 依赖的 ceil(x,y) 宏在此定义
#include "pad.hpp"

using namespace AscendC;

namespace sgetri_p2 {

__aicore__ inline void p2_mte3_mte2()
{
    int32_t ev = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
}

__aicore__ inline void p2_mte2_v()
{
    int32_t ev = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
}

__aicore__ inline void p2_v_mte3()
{
    int32_t ev = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev);
}

/*
 * R1: 单核全量版 eye/pad/restore —— 直接调 pad.hpp 类的全量接口,
 * 不走 *Split 分片、不发 CrossCore flag (写入范围与主路径语义一致)。
 */
__aicore__ inline void rebuild_eye_real_p2(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN,
                                           GM_ADDR eye_gm)
{
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;
    int totalRows = (M + 128) * 3;

    GlobalTensor<float> eyeGlobal;
    eyeGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eye_gm));

    EyeReal op;
    op.Init(&tbufPool, eyeGlobal, orgN, strideN);
    op.BuildRows(0, totalRows);
    tbufPool.Reset();
}

__aicore__ inline void pad_real_p2(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, GM_ADDR A_org,
                                   GM_ADDR A_work)
{
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
    op.RealWork(0, 0, orgM);
    op.ImagZeroWork(0, orgM);
    tbufPool.Reset();
}

__aicore__ inline void restore_real_p2(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, GM_ADDR A_inv,
                                       GM_ADDR eye_gm)
{
    int strideN = (orgN + 127) / 128 * 128;

    GlobalTensor<float> aGlobalInv;
    GlobalTensor<float> eyeGlobalReal;
    aGlobalInv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_inv));
    eyeGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eye_gm));

    RestoreReal op;
    op.Init(&tbufPool, eyeGlobalReal, aGlobalInv, orgM, orgN, strideN);
    op.Process(0, 0, orgM);
    tbufPool.Reset();
}

/*
 * R2: 行交换 (主路径 SwapRows 职责, 单核小容量版)。
 * 段长 <= N2 <= 256, 512 元素队列足够。
 */
template <typename T>
class SwapRowsP2 {
    static constexpr int TILE_LENGTH = 512;
    GlobalTensor<T> aGlobalReal, aGlobalImag, eyeGlobal;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind1;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind2;

public:
    __aicore__ inline SwapRowsP2() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<T> aGlobalReal,
                                GlobalTensor<T> aGlobalImag, GlobalTensor<T> eyeGlobal)
    {
        pipe->InitBuffer(queBind1, 1, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(queBind2, 1, TILE_LENGTH * sizeof(T));
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->eyeGlobal = eyeGlobal;
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int N)
    {
        if (offsetA == offsetB || N <= 0) {
            return;
        }
        ProcessKernel(offsetA, offsetB, N, aGlobalReal);
        ProcessKernel(offsetA, offsetB, N, aGlobalImag);
    }
    __aicore__ inline void ProcessKernel(int offsetA, int offsetB, int N, GlobalTensor<T> aGlobal)
    {
        LocalTensor<T> bindLocal1 = queBind1.AllocTensor<T>();
        LocalTensor<T> bindLocal2 = queBind2.AllocTensor<T>();
        DataCopy(bindLocal1, aGlobal[offsetA], N);
        DataCopy(bindLocal2, aGlobal[offsetB], N);
        queBind1.EnQue(bindLocal1);
        queBind2.EnQue(bindLocal2);
        int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

        queBind1.DeQue<T>();
        queBind2.DeQue<T>();
        DataCopy(aGlobal[offsetB], bindLocal1, N);
        DataCopy(aGlobal[offsetA], bindLocal2, N);
        queBind1.FreeTensor(bindLocal1);
        queBind2.FreeTensor(bindLocal2);

        int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
    __aicore__ inline void ProcessEye(int offsetA, int offsetB, int N)
    {
        if (offsetA == offsetB || N <= 0) {
            return;
        }
        LocalTensor<T> bindLocal1 = queBind1.AllocTensor<T>();
        LocalTensor<T> bindLocal2 = queBind2.AllocTensor<T>();
        DataCopy(bindLocal1, eyeGlobal[offsetA], N);
        DataCopy(bindLocal2, eyeGlobal[offsetB], N);
        queBind1.EnQue(bindLocal1);
        queBind2.EnQue(bindLocal2);
        int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

        bindLocal1 = queBind1.DeQue<T>();
        bindLocal2 = queBind2.DeQue<T>();
        DataCopy(eyeGlobal[offsetB], bindLocal1, N);
        DataCopy(eyeGlobal[offsetA], bindLocal2, N);
        queBind1.FreeTensor(bindLocal1);
        queBind2.FreeTensor(bindLocal2);

        int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
};

/*
 * 16xW 实数面板原位求解 (R3/R6):
 *   lower:  X[r] -= sum_{p<r} L[r,p]*X[p]           (L 单位对角)
 *   upper:  X[r] = (X[r] - sum_{p>r} U[r,p]*X[p]) / U[r,r]
 *           (r >= mask 的行输出置 0 —— orgM 尾部 padding 行, U 对角为 0)
 * X: xG 行 [p0,p0+16) 列 [colStart,colStart+W), 行距 strideN;
 * L/U 面板: lG[p0*strideN+p0] 起 16x16, 行距 strideN。
 * 输出正号直接写回 (P2 自有约定; Schur/尾端系数相应取 -c)。
 * CopyIn/Compute/CopyOut 三段式 + 逐行 DataCopyPad (pad.hpp 范式;
 * 避免 DataCopyExtParams 多行 stride 语义歧义)。
 */
class PanelSolve {
    int strideN;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECIN, 1> lQue;

public:
    __aicore__ inline PanelSolve() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, int strideN)
    {
        this->strideN = strideN;
        pipe->InitBuffer(xQue, 1, 16 * 256 * sizeof(float));
        pipe->InitBuffer(lQue, 1, 16 * 16 * sizeof(float));
    }

    __aicore__ inline void Process(GlobalTensor<float> xG, GlobalTensor<float> lG, int p0, int colStart, int W,
                                   bool upper, int mask)
    {
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(W * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pp{false, 0, 0, 0};
        DataCopyExtParams rowParamsL{1, 16 * sizeof(float), 0, 0, 0};

        LocalTensor<float> x;
        LocalTensor<float> l;
        {
            x = xQue.AllocTensor<float>();
            for (int r = 0; r < 16; ++r) {
                DataCopyPad(x[r * W], xG[(p0 + r) * strideN + colStart], rowParams, pp);
            }
            xQue.EnQue(x);
        }
        {
            l = lQue.AllocTensor<float>();
            for (int r = 0; r < 16; ++r) {
                DataCopyPad(l[r * 16], lG[(p0 + r) * strideN + p0], rowParamsL, pp);
            }
            lQue.EnQue(l);
        }
        x = xQue.DeQue<float>();
        l = lQue.DeQue<float>();
        PipeBarrier<PIPE_ALL>(); // MTE2 写 -> S 标量读 (GetValue) 全序

        if (!upper) {
            for (int r = 0; r < 16; ++r) {
                for (int p = 0; p < r; ++p) {
                    float s = l.GetValue(r * 16 + p);
                    if (s == 0.0f) {
                        continue;
                    }
                    Axpy(x[r * W], x[p * W], -s, W);
                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            for (int r = 15; r >= 0; --r) {
                if (r >= mask) {
                    Duplicate(x[r * W], (float)0.0, W);
                    PipeBarrier<PIPE_V>();
                    continue;
                }
                for (int p = r + 1; p < 16; ++p) {
                    float s = l.GetValue(r * 16 + p);
                    if (s == 0.0f) {
                        continue;
                    }
                    Axpy(x[r * W], x[p * W], -s, W);
                    PipeBarrier<PIPE_V>();
                }
                float d = l.GetValue(r * 16 + r);
                float inv = 1.0f / d;
                PipeBarrier<PIPE_ALL>();
                Muls(x[r * W], x[r * W], inv, W);
                PipeBarrier<PIPE_V>();
            }
        }

        // R11v2: 裸 MTE3 写回, 但 FreeTensor 必须在 MTE3 读完成之后 ——
        // VECIN 队列 Free 不跟踪 MTE3 读依赖, 提前 Free 会让下一轮
        // Alloc+MTE2 复写与在飞 MTE3 读竞争同一 UB buffer (间歇性错根源)
        p2_v_mte3(); // V 写完 -> MTE3 可写
        for (int r = 0; r < 16; ++r) {
            DataCopyPad(xG[(p0 + r) * strideN + colStart], x[r * W], rowParams);
        }
        p2_mte3_mte2(); // MTE3 读 x 完成 -> 才允许 Free/复写
        xQue.FreeTensor(x);
        lQue.FreeTensor(l);
    }
};

/*
 * AIV 版尾端更新 (R4/R10):
 *   Lower: X[i, cs:ce] -= tri[i,p] * X[p, cs:ce], i in (p0+pn, M)
 *   Upper: X[i, cs:ce] -= tri[i,p] * X[p, cs:ce], i in [0, p0)
 * 用途: LU 阶段 Schur (tri=L, x=A_work, 列 [p0+16,N2)) 与求解阶段尾端
 * (tri=L/U, x=eye real, 列 [0,N2))。
 * 面板行常驻 TBuf (LoadPanel 显式 MTE2_V); 行块与系数块各走独立
 * VECIN 队列; 计算后 V->MTE3 事件 + 直接 MTE3 写回 (LUCustom1 CopyOut
 * 同款), tile 末 MTE3->MTE2 事件保护下一 tile 的 MTE2 复写。
 * 系数块: 每 tile 搬 [r0,r0+rt)x[p0,p0+pn) 入 UB (R10, 避免 ~n^2 次
 * GM 标量读); tile 行数封顶 128 (coeffQue 8KB)。
 */
class TailUpdate {
    static constexpr int PANEL = 16;
    static constexpr int MAX_ROWS = 128;
    GlobalTensor<float> triG; // 三角因子系数 plane
    GlobalTensor<float> xG;   // 源/目的 plane (原位更新)
    int M;
    int strideN;
    int tileEle;
    TBuf<TPosition::VECCALC> panelBuf;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECIN, 1> coeffQue;

public:
    __aicore__ inline TailUpdate() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<float> tri, GlobalTensor<float> x,
                                int M, int strideN, int tileEle)
    {
        this->triG = tri;
        this->xG = x;
        this->M = M;
        this->strideN = strideN;
        this->tileEle = tileEle;
        pipe->InitBuffer(panelBuf, PANEL * strideN * sizeof(float));
        pipe->InitBuffer(inQue, 1, tileEle * sizeof(float));
        pipe->InitBuffer(coeffQue, 1, MAX_ROWS * 16 * sizeof(float));
    }

    /* 面板解行 [p0,p0+pn) 列 [cs,cs+w) 载入常驻 TBuf */
    __aicore__ inline void LoadPanel(int p0, int pn, int cs, int w, LocalTensor<float>& panel)
    {
        PipeBarrier<PIPE_ALL>(); // 上一轮 panel 的 V 读 -> 本轮 MTE2 写 (WAR)
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(w * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pp{false, 0, 0, 0};
        for (int r = 0; r < pn; ++r) {
            DataCopyPad(panel[r * w], xG[(p0 + r) * strideN + cs], rowParams, pp);
        }
        p2_mte2_v();
    }

    /* 行块 [r0,r0+rt): 系数+数据载入 -> 逐 (i,p) Axpy -> 写回 */
    __aicore__ inline void TileStep(int r0, int rt, int p0, int pn, int cs, int w, LocalTensor<float>& panel)
    {
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(w * sizeof(float)), 0, 0, 0};
        DataCopyExtParams rowParamsC{1, static_cast<uint32_t>(pn * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pp{false, 0, 0, 0};

        LocalTensor<float> coeff;
        LocalTensor<float> src;
        { // 系数块 [r0,r0+rt) x [p0,p0+pn) -> UB
            coeff = coeffQue.AllocTensor<float>();
            for (int r = 0; r < rt; ++r) {
                DataCopyPad(coeff[r * pn], triG[(r0 + r) * strideN + p0], rowParamsC, pp);
            }
            coeffQue.EnQue(coeff);
        }
        { // 数据块 [r0,r0+rt) x [cs,cs+w) -> UB
            src = inQue.AllocTensor<float>();
            for (int r = 0; r < rt; ++r) {
                DataCopyPad(src[r * w], xG[(r0 + r) * strideN + cs], rowParams, pp);
            }
            inQue.EnQue(src);
        }
        coeff = coeffQue.DeQue<float>();
        src = inQue.DeQue<float>();
        PipeBarrier<PIPE_ALL>(); // MTE2 写 -> S 标量读全序

        for (int ii = 0; ii < rt; ++ii) {
            for (int p = p0; p < p0 + pn; ++p) {
                float c = coeff.GetValue(ii * pn + (p - p0));
                if (c == 0.0f) {
                    continue;
                }
                Axpy(src[ii * w], panel[(p - p0) * w], -c, w);
                PipeBarrier<PIPE_V>();
            }
        }

        // R11v2: MTE3 读完成事件必须先于 FreeTensor (同 PanelSolve)
        p2_v_mte3(); // V 写完 -> MTE3 可写
        for (int r = 0; r < rt; ++r) {
            DataCopyPad(xG[(r0 + r) * strideN + cs], src[r * w], rowParams);
        }
        p2_mte3_mte2(); // MTE3 读 src 完成 -> 才允许 Free/下一 tile 复写
        inQue.FreeTensor(src);
        coeffQue.FreeTensor(coeff);
    }

    /* 下三角尾端/Schur: 更新行 [p0+pn, M) 列 [cs,ce) */
    __aicore__ inline void Lower(int p0, int pn, int cs, int ce)
    {
        int w = ce - cs;
        int rowsPerTile = min(min(tileEle / w, MAX_ROWS), M - p0 - pn);
        if (rowsPerTile < 1) {
            rowsPerTile = 1;
        }
        LocalTensor<float> panel = panelBuf.Get<float>();
        LoadPanel(p0, pn, cs, w, panel);
        for (int r0 = p0 + pn; r0 < M; r0 += rowsPerTile) {
            int rt = min(rowsPerTile, M - r0);
            TileStep(r0, rt, p0, pn, cs, w, panel);
        }
    }

    /* 上三角尾端 (自底向上): 更新行 [0, p0) 列 [cs,ce) */
    __aicore__ inline void Upper(int p0, int pn, int cs, int ce)
    {
        int w = ce - cs;
        int rowsPerTile = min(min(tileEle / w, MAX_ROWS), p0);
        if (rowsPerTile < 1) {
            rowsPerTile = 1;
        }
        LocalTensor<float> panel = panelBuf.Get<float>();
        LoadPanel(p0, pn, cs, w, panel);
        for (int rEnd = p0; rEnd > 0; rEnd -= rowsPerTile) {
            int r0 = max(0, rEnd - rowsPerTile);
            int rt = rEnd - r0;
            TileStep(r0, rt, p0, pn, cs, w, panel);
        }
    }
};

} // namespace sgetri_p2

/*
 * P2 单矩阵全流程 (一个 AIV 核):
 * eye -> pad -> [逐面板: LU面板 -> W行交换 -> U12 -> Schur]
 * -> 下三角求解 -> 上三角求解 -> restore
 */
__aicore__ inline void custom_getri_per_core(int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org,
                                             GM_ADDR A_inv, GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm,
                                             GM_ADDR gather1_gm, GM_ADDR gather2_gm, GM_ADDR eye_gm, GM_ADDR workspace)
{
    using namespace sgetri_p2;
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;
    int N2 = (orgN + 15) / 16 * 16;
    int workM = (M + 511) / 512 * 512;
    (void)blockM;
    (void)blockN;
    (void)tileM;
    (void)workspace;

    TPipe pipe;
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    pipe.InitBufPool(tbufPool, 192 * 1024);

    // 1) eye 重建 + pad (R1: 单核全量版, 无 CrossCore barrier)
    rebuild_eye_real_p2(tbufPool, orgM, orgN, eye_gm);
    pad_real_p2(tbufPool, orgM, orgN, A_org, A_work);
    p2_mte3_mte2(); // R8: pad 写回(MTE3) -> LU 读(MTE2)

    // 2) LU: 逐 16 列面板 (transpose 进出 work 区 + LUCustom1),
    //    面板后单核补做 pivot 行交换 (R2) + U12 求解 + Schur 更新 (R3)
    {
        GlobalTensor<float> aReal;
        GlobalTensor<float> aImag;
        GlobalTensor<float> workG;
        GlobalTensor<float> workImagG;
        GlobalTensor<float> eyeG;
        GlobalTensor<uint32_t> wG;
        GlobalTensor<uint32_t> g1;
        GlobalTensor<uint32_t> g2;
        aReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_work));
        aImag = aReal[strideN * M];
        workG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work_gm));
        workImagG = workG[workM * 16]; // 命名变量: Init 收非const引用
        eyeG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eye_gm));
        wG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(W));
        g1.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gather1_gm));
        g2.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gather2_gm));

        TQue<QuePosition::VECIN, 1> q1;
        TQue<QuePosition::VECIN, 1> q2;
        TQue<QuePosition::VECOUT, 1> q3;
        TQue<QuePosition::VECIN, 1> wInQue; // 根因 A 修复: swap 读 W 用
        TransPose<float> optr;
        LUCustom1<float> oplu;
        SwapRowsP2<float> opswap;
        PanelSolve pslv;
        TailUpdate tail;

        int cnt = M / 16;
        for (int i = 0; i < cnt; ++i) {
            int offsetN = i * 16;
            int realN = min(orgN - offsetN, 16);
            int realM = min(orgM - offsetN, 16);
            int blockM_i = M - offsetN;
            if (i == 0) {
                // R5: 仅此一次 Reset+Init (阶段内循环不再 Reset)
                tbufPool.Reset();
                tbufPool.InitBuffer(q1, 1, 8192 * sizeof(float));
                tbufPool.InitBuffer(q2, 1, 8192 * sizeof(float));
                tbufPool.InitBuffer(q3, 1, 8192 * sizeof(float));
                tbufPool.InitBuffer(wInQue, 1, 512 * sizeof(uint32_t)); // 16 条/面板足够
                optr.Init(&tbufPool, q1, q2, q3, aReal, aImag, workG, workImagG, g1, g2, workM, strideN, 512, 16);
                oplu.Init(&tbufPool, q1, q2, workG, workImagG, wG, workM, 16, 16);
                oplu.SetP2Mode(&tbufPool); // 根因 A 修复: W 写 VECOUT 队列化
                opswap.Init(&tbufPool, aReal, aImag, eyeG);
                pslv.Init(&tbufPool, strideN);
                tail.Init(&tbufPool, aReal, aReal, M, strideN, 2048); // Schur 小 tile 省 UB
            }

            // a) 分解面板 i (transpose -> LUCustom1 -> transpose 回)
            int srcOffset = offsetN * strideN + offsetN; // R7
            optr.Process1(srcOffset, 0, blockM_i);
            p2_mte3_mte2();
            oplu.Process(512 - blockM_i, offsetN, offsetN, blockM_i, blockM_i + orgM - M, realN);
            p2_mte3_mte2();
            optr.Process2(srcOffset, 0, blockM_i);
            p2_mte3_mte2(); // transpose 回写(MTE3) -> 行交换读(MTE2)

            // b) 应用面板 i 的 pivot 行交换（对齐主路径 lu_custom.hpp 时序：
            //    上一面板 swap 先于本段 U12/Schur 执行, swap 写(MTE3) ->
            //    U12 读(MTE2) 经下方屏障衔接）

            // c) 应用面板 i 的 pivot 行交换 (主路径 block8/block10 职责):
            //    A 尾列段 [offsetN+16,N2)（面板列 [offsetN,offsetN+16) 已在
            //    LUCustom1 内部 getf2 物理换行, 不重复 swap —— 原实现从
            //    offsetN 开始, 面板列被双重置换, L/U 因子与 W 行序错位,
            //    issue #301）+ L 列段 [0,offsetN) + eye 全行
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(wG[offsetN]);
            // 读取边界契约: steps = min(realM, realN) 与 LUCustom1 本面板
            // 写入条数 t = min(realM, realN) (getf2.hpp, 调用参数同为
            // orgM-offsetN / min(orgN-offsetN,16)) 严格一致, 即只读
            // [offsetN, offsetN+steps) 内已写入条目, 不触碰面板 padding 段
            // 与 host 初值哨兵 (0xFFFFFFFF)。
            int steps = min(realM, realN);
            // 根因 A 修复 (读侧): W 面板段经 VECIN 队列 MTE2 搬入 UB 后标量
            // 读 —— MTE2 读 GM 见的是 LUCustom1 写侧 VECOUT MTE3 已落地的数
            // 据, 队列仲裁 + MTE3->MTE2 (写侧尾事件) 保证当轮可见。
            if (steps > 0) {
                LocalTensor<uint32_t> wIn = wInQue.AllocTensor<uint32_t>();
                DataCopy(wIn, wG[offsetN], steps);
                wInQue.EnQue(wIn);
                wIn = wInQue.DeQue<uint32_t>();
                PipeBarrier<PIPE_ALL>();
                for (int j = offsetN; j < offsetN + steps; ++j) {
                    int p = static_cast<int>(wIn.GetValue(j - offsetN));
                    // A 尾列段基址 offsetN+16（面板右边界, 对齐主路径
                    // lu_custom.hpp:225 的 offset）—— 尾列为面板右侧全部列;
                    // 末面板无尾列段时跳过（L 列段与 eye 段仍需交换）
                    if (offsetN + 16 < N2) {
                        opswap.Process(offsetN + 16 + j * strideN, offsetN + 16 + p * strideN, N2 - offsetN - 16);
                    }
                    opswap.Process(j * strideN, p * strideN, offsetN);
                    opswap.ProcessEye(j * strideN, p * strideN, N2);
                }
                PipeBarrier<PIPE_ALL>();
                wInQue.FreeTensor(wIn);
            }
            p2_mte3_mte2(); // swap 写(MTE3) -> U12 读(MTE2)

            // d) U12 = Lpanel^-1 * A[面板行, 尾列] (正号)
            //    + Schur: A[下方行, 尾列] -= L[i,p] * U12[p, :]
            //    （基于已 swap 的尾列数据；主路径同为 swap -> trsm/Schur 时序）
            if (offsetN + 16 < N2) {
                int w = N2 - offsetN - 16;
                pslv.Process(aReal, aReal, offsetN, offsetN + 16, w, false, 16);
                p2_mte3_mte2(); // U12 写回(MTE3) -> Schur 读(MTE2)
                tail.Lower(offsetN, 16, offsetN + 16, N2);
                p2_mte3_mte2(); // Schur 写回(MTE3) -> 下一面板 Process1 读(MTE2)
            }
        }
        p2_mte3_mte2();
        tbufPool.Reset();
    }

    // 3) 三角求解 (X = eye): 下三角自顶向下 + 上三角自底向上
    //    每面板: PanelSolve 解 16x16 面板系 + TailUpdate 尾端更新
    {
        GlobalTensor<float> aReal;
        GlobalTensor<float> eyeG;
        aReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_work));
        eyeG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eye_gm));

        int cnt = M / 16;
        {
            PanelSolve pslv;
            TailUpdate tail;
            pslv.Init(&tbufPool, strideN);
            tail.Init(&tbufPool, aReal, eyeG, M, strideN, 8192);
            for (int pi = 0; pi < cnt; ++pi) {
                int p0 = pi * 16;
                // 面板内单位下三角求解: Y[p0:p0+16] = Lblk^-1 * B。
                // 注意: 面板内求解与跨面板 tail 更新缺一不可 —— 仅做 tail
                // 会让面板内 L_strict 项全部丢失, cnt>=2 (n>=32) 起误差
                // 1e-4~1e-3 量级 (n<32 不经 sgetri 路由, 用例不暴露)。
                pslv.Process(eyeG, aReal, p0, 0, N2, false, 16);
                p2_mte3_mte2();
                if (pi + 1 < cnt) {
                    tail.Lower(p0, 16, 0, N2);
                    p2_mte3_mte2();
                }
            }
        }
        p2_mte3_mte2();
        tbufPool.Reset();

        {
            PanelSolve pslv;
            TailUpdate tail;
            pslv.Init(&tbufPool, strideN);
            tail.Init(&tbufPool, aReal, eyeG, M, strideN, 8192);
            for (int pi = cnt - 1; pi >= 0; --pi) {
                int p0 = pi * 16;
                // mask: 末面板有效行数 (orgM 尾部), 与主路径 custom_trsm_upper 一致
                int mask = (pi == cnt - 1) ? (orgM % 16 ? orgM % 16 : 16) : 16;
                pslv.Process(eyeG, aReal, p0, 0, N2, true, mask);
                p2_mte3_mte2();
                if (pi > 0) {
                    tail.Upper(p0, 16, 0, N2);
                    p2_mte3_mte2();
                }
            }
        }
        p2_mte3_mte2();
        tbufPool.Reset();
    }

    // 4) restore: eye real 段 -> A_inv (R1: 单核全量版)
    restore_real_p2(tbufPool, orgM, orgN, A_inv, eye_gm);
    p2_mte3_mte2();

    pipe.Destroy();
}

#endif // _GETRI_PER_CORE_HPP_
