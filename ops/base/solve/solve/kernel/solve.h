/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_SOLVE_AIV_H
#define ASDSIP_SOLVE_AIV_H

#include <cstdint>
#include "kernel_operator.h"
#include "../tiling/solve_tiling_data.h"

using namespace AscendC;

namespace SolveOp {

constexpr uint32_t MAX_N = AsdSip::SOLVE_RESIDENT_MAX_N;              // 整装 LU 路径的阶数上限
constexpr uint32_t BIG_N = AsdSip::SOLVE_MAX_N;                       // 总阶数上限(分块 LU 路径)
constexpr uint32_t PANEL_W = 64;                                      // 分块 LU 的 panel 宽(= 1 repeat)
constexpr uint32_t ROW_STRIDE = AsdSip::SOLVE_ROW_STRIDE;             // A 每行物理跨度(1 repeat = 64 floats)
constexpr uint32_t REPEAT_ELEMS = 64;                                 // fp32 每 repeat 元素数
constexpr uint32_t MAX_COLS_PER_BLK = AsdSip::SOLVE_MAX_COLS_PER_BLK; // 每块列数上限(bcastRow/mulOut 容量)
constexpr uint32_t MAX_B_BLK_ELEMS = AsdSip::SOLVE_MAX_B_BLK_ELEMS;   // B 块缓冲容量(整装路径)
constexpr uint32_t MIN_N2_CHUNK = AsdSip::SOLVE_MIN_N2_CHUNK;         // N2 路径最小聚合批量

constexpr uint32_t N2_VEC_GROUP = 64; // n=2 小 cols 路径: 64 矩阵打包 64 车道
constexpr uint32_t N2_VEC_MAX_COLS = 1;

// 求解面板分块: 系数批量驻留(借鉴 sgetri TailUpdate)
constexpr uint32_t SOLVE_PBLK_W = 48;  // 求解面板宽(面板块 PBLK_W² + 尾端条 TILE_R×PBLK_W 驻留)
constexpr uint32_t SOLVE_TILE_R = 128; // 尾端条行数

using AsdSip::SolveN2ChunkCap;

// 求解 A·X = B (fp32/ND, batch 个小矩阵): LU + 前代/回代, 消元因子存 L, 每列块重放
// 任务划分: useN2=1 → ProcessN2 矩阵均分; 否则全局任务 g=m*colBlkNum+b 均分(两核差≤1)
// 同步: 管间 SetFlag/WaitFlag, 标量↔向量交界 pipe_barrier(PIPE_ALL)
class SolveAIV {
public:
    __aicore__ inline SolveAIV(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReadTiling(GM_ADDR tilingGm);
    __aicore__ inline void LoadA(uint64_t batchIdx);
    __aicore__ inline void SwapRows(uint32_t r1, uint32_t r2);
    __aicore__ inline void DecomposeLu();
    __aicore__ inline void LoadBPerm(uint64_t batchIdx, uint64_t blkIdx, uint32_t blkCols);
    __aicore__ inline void FwdSub(uint32_t vecWidth);
    __aicore__ inline void BackSub(uint32_t vecWidth);
    __aicore__ inline void StoreX(uint64_t batchIdx, uint64_t blkIdx, uint32_t blkCols);
    __aicore__ inline void ProcessN2();
    __aicore__ inline void ProcessN2Vec(uint64_t base, uint32_t num);
    __aicore__ inline void ProcessLu();
    __aicore__ inline void GMSwapSeg(uint64_t wsBase, uint32_t rowA, uint32_t rowB, uint32_t c0, uint32_t c1,
                                     LocalTensor<float> t1, LocalTensor<float> t2);
    __aicore__ inline void ProcessLuBig();

    TPipe pipe;
    TBuf<QuePosition::VECCALC> uBuf;

    GlobalTensor<float> aGM;
    GlobalTensor<float> bGM;
    GlobalTensor<float> xGM;
    GlobalTensor<float> wsGM; // n>64: A 副本(in-place 分解, 不破坏输入)

    LocalTensor<float> matU;     // [MAX_N][ROW_STRIDE] 工作矩阵, 分解后上三角=U
    LocalTensor<float> matL;     // [MAX_N][ROW_STRIDE] 严格下三角=L, 行尾存 perm
    LocalTensor<float> luTmp;    // [ROW_STRIDE] 行临时
    LocalTensor<float> bcastRow; // [MAX_COLS_PER_BLK] 广播行
    LocalTensor<float> mulOut;   // [MAX_COLS_PER_BLK] 乘积输出行
    LocalTensor<float> bBlk;     // [n][colsPerBlk] B 列块(perm 重排载入, 前代后存 y)
    LocalTensor<float> xBlk;     // [n][colsPerBlk] X 列块(回代输出)

    uint32_t n = 0;
    uint32_t cols = 0;
    uint32_t colsPerBlk = 0;
    uint32_t colBlkNum = 0;
    uint32_t useN2 = 0;     // 1=批量标量路径(host 判定)
    uint32_t taskFirst = 0; // LU 路径: 本核全局任务段 [taskFirst, taskFirst+taskCnt)
    uint32_t taskCnt = 0;
    uint32_t batchStart = 0; // N2 路径: 本核 batch 段 [batchStart, batchStart+batchNum)
    uint32_t batchNum = 0;
};

// 按 SolveTilingData 结构体访问 tiling GM
__aicore__ inline void SolveAIV::ReadTiling(GM_ADDR tilingGm)
{
    auto tiling = reinterpret_cast<__gm__ AsdSip::SolveTilingData*>(tilingGm);
    this->n = tiling->n;
    this->cols = tiling->cols;
    this->colsPerBlk = tiling->colsPerBlk;
    this->colBlkNum = tiling->colBlkNum;
    this->useN2 = tiling->useN2;
    uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    this->taskFirst = tiling->taskStart[coreIdx];
    this->taskCnt = tiling->taskNum[coreIdx];
    this->batchStart = tiling->batchStart[coreIdx];
    this->batchNum = tiling->batchNum[coreIdx];
}

__aicore__ inline void SolveAIV::Init(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->wsGM.SetGlobalBuffer((__gm__ float*)workSpace);
    SetMaskNorm();
    SetAtomicNone();

    ReadTiling(tilingGm);

    this->aGM.SetGlobalBuffer((__gm__ float*)a);
    this->bGM.SetGlobalBuffer((__gm__ float*)b);
    this->xGM.SetGlobalBuffer((__gm__ float*)x);

    this->pipe.InitBuffer(
        this->uBuf, (MAX_N * ROW_STRIDE * 2 + ROW_STRIDE + MAX_COLS_PER_BLK * 2 + MAX_B_BLK_ELEMS * 2) * sizeof(float));
    LocalTensor<float> ubufLocal = uBuf.Get<float>();
    uint32_t off = 0;
    this->matU = ubufLocal[off];
    off += MAX_N * ROW_STRIDE;
    this->matL = ubufLocal[off];
    off += MAX_N * ROW_STRIDE;
    this->luTmp = ubufLocal[off];
    off += ROW_STRIDE;
    this->bcastRow = ubufLocal[off];
    off += MAX_COLS_PER_BLK;
    this->mulOut = ubufLocal[off];
    off += MAX_COLS_PER_BLK;
    this->bBlk = ubufLocal[off];
    off += MAX_B_BLK_ELEMS;
    this->xBlk = ubufLocal[off];

    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void SolveAIV::LoadA(uint64_t batchIdx)
{
    auto eventId = EVENT_ID0;
    // V 清零后再 MTE2 拷贝, 避免清零覆盖
    Duplicate<float>(matU, 0, MAX_N * ROW_STRIDE);
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    for (uint32_t r = 0; r < this->n; r++) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->n * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(matU[r * ROW_STRIDE], aGM[static_cast<uint64_t>(batchIdx) * this->n * this->n + r * this->n],
                    copyParams, padParams);
    }
}

__aicore__ inline void SolveAIV::SwapRows(uint32_t r1, uint32_t r2)
{
    // 整行向量交换(64 float 整块对齐); matL 行尾 perm 随行更新; luTmp 复用为行暂存
    Copy(luTmp, matU[r1 * ROW_STRIDE], ROW_STRIDE, 1, {1, 1, 8, 8});
    pipe_barrier(PIPE_V);
    Copy(matU[r1 * ROW_STRIDE], matU[r2 * ROW_STRIDE], ROW_STRIDE, 1, {1, 1, 8, 8});
    pipe_barrier(PIPE_V);
    Copy(matU[r2 * ROW_STRIDE], luTmp, ROW_STRIDE, 1, {1, 1, 8, 8});
    pipe_barrier(PIPE_V); // luTmp 切换为 L 行暂存
    Copy(luTmp, matL[r1 * ROW_STRIDE], ROW_STRIDE, 1, {1, 1, 8, 8});
    pipe_barrier(PIPE_V);
    Copy(matL[r1 * ROW_STRIDE], matL[r2 * ROW_STRIDE], ROW_STRIDE, 1, {1, 1, 8, 8});
    pipe_barrier(PIPE_V);
    Copy(matL[r2 * ROW_STRIDE], luTmp, ROW_STRIDE, 1, {1, 1, 8, 8});
    pipe_barrier(PIPE_V);
}

// 带部分主元 LU: matU 上三角=U(下三角恒 0, 整行消元安全), matL 下三角=L, 行尾=perm
__aicore__ inline void SolveAIV::DecomposeLu()
{
    for (uint32_t r = 0; r < this->n; r++) {
        matL.SetValue(r * ROW_STRIDE + (ROW_STRIDE - 1), static_cast<float>(static_cast<int32_t>(r)));
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t k = 0; k < this->n; k++) {
        pipe_barrier(PIPE_ALL);
        uint32_t pivot = k;
        float bestAbs = matU.GetValue(k * ROW_STRIDE + k);
        bestAbs = bestAbs < 0 ? -bestAbs : bestAbs;
        for (uint32_t r = k + 1; r < this->n; r++) {
            float v = matU.GetValue(r * ROW_STRIDE + k);
            v = v < 0 ? -v : v;
            if (v > bestAbs) {
                bestAbs = v;
                pivot = r;
            }
        }
        if (pivot != k) {
            SwapRows(pivot, k);
            pipe_barrier(PIPE_ALL);
        }
        float piv = matU.GetValue(k * ROW_STRIDE + k);
        float ipiv = 1.0f / piv;
        for (uint32_t r = k + 1; r < this->n; r++) {
            float lFactor = matU.GetValue(r * ROW_STRIDE + k) * ipiv;
            pipe_barrier(PIPE_ALL);
            // Axpy 融合(单舍入, 乘积全精度): 替代 Duplicate+Mul+Sub; j<k 列恒 0, 全宽 64 语义不变
            Axpy(matU[r * ROW_STRIDE], matU[k * ROW_STRIDE], -lFactor, ROW_STRIDE, 1, {1, 1, 8, 8});
            pipe_barrier(PIPE_V);
            matL.SetValue(r * ROW_STRIDE + k, lFactor); // L 存 matL(该位消元后为 0)
        }
    }
    pipe_barrier(PIPE_ALL);
}

// bBlk[r] ← B 源行 perm[r](按分解行置换重排)
__aicore__ inline void SolveAIV::LoadBPerm(uint64_t batchIdx, uint64_t blkIdx, uint32_t blkCols)
{
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    for (uint32_t r = 0; r < this->n; r++) {
        uint32_t src = static_cast<uint32_t>(static_cast<int32_t>(matL.GetValue(r * ROW_STRIDE + (ROW_STRIDE - 1))));
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(blkCols * sizeof(float)), 0, 0, 0};
        DataCopyPad(bBlk[r * this->colsPerBlk],
                    bGM[(static_cast<uint64_t>(batchIdx) * this->n + src) * this->cols + blkIdx * this->colsPerBlk],
                    copyParams, padParams);
    }
}

// 前代 L·y = P·B: bBlk 原地变 y(单位下三角, 列向量化)
__aicore__ inline void SolveAIV::FwdSub(uint32_t vecWidth)
{
    uint8_t rep = static_cast<uint8_t>(vecWidth / REPEAT_ELEMS);
    pipe_barrier(PIPE_ALL);
    for (uint32_t k = 0; k < this->n; k++) {
        for (uint32_t j = 0; j < k; j++) {
            float lCoef = matL.GetValue(k * ROW_STRIDE + j);
            pipe_barrier(PIPE_ALL);
            Axpy(bBlk[k * this->colsPerBlk], bBlk[j * this->colsPerBlk], -lCoef, REPEAT_ELEMS, rep, {1, 1, 8, 8});
            pipe_barrier(PIPE_V);
        }
    }
    pipe_barrier(PIPE_ALL);
}

// 回代 U·x = y: 读 bBlk(y)/matU, 写 xBlk; y 原地累减后乘 1/U[k][k]
__aicore__ inline void SolveAIV::BackSub(uint32_t vecWidth)
{
    uint8_t rep = static_cast<uint8_t>(vecWidth / REPEAT_ELEMS);
    pipe_barrier(PIPE_ALL);
    for (int32_t k = static_cast<int32_t>(this->n) - 1; k >= 0; k--) {
        uint32_t kr = static_cast<uint32_t>(k);
        for (uint32_t j = kr + 1; j < this->n; j++) {
            float uCoef = matU.GetValue(kr * ROW_STRIDE + j);
            pipe_barrier(PIPE_ALL);
            Axpy(bBlk[kr * this->colsPerBlk], xBlk[j * this->colsPerBlk], -uCoef, REPEAT_ELEMS, rep, {1, 1, 8, 8});
            pipe_barrier(PIPE_V);
        }
        float ipiv = 1.0f / matU.GetValue(kr * ROW_STRIDE + kr);
        pipe_barrier(PIPE_ALL);
        Duplicate<float>(bcastRow, ipiv, vecWidth);
        pipe_barrier(PIPE_V);
        Mul(xBlk[kr * this->colsPerBlk], bcastRow, bBlk[kr * this->colsPerBlk], REPEAT_ELEMS, rep, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
    }
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void SolveAIV::StoreX(uint64_t batchIdx, uint64_t blkIdx, uint32_t blkCols)
{
    for (uint32_t j = 0; j < this->n; j++) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(blkCols * sizeof(float)), 0, 0, 0};
        DataCopyPad(xGM[(static_cast<uint64_t>(batchIdx) * this->n + j) * this->cols + blkIdx * this->colsPerBlk],
                    xBlk[j * this->colsPerBlk], copyParams);
    }
}

// n=2 小 cols 向量路径(借鉴 smatinv_batched): 64 矩阵打包 64 车道, 单条向量指令推进
//   - A/B 整组连续搬入, V 管 Gather 拆包; 无 Scatter, 每列 2 条 MTE3 散射写
//   - 主元判定平方免 Abs(a00²≥a10² ⟺ |a00|≥|a10|)
//   - 上/下主元 Compare+Select 混合角色, 单套公式无分支(与标量路径逐式同构)
//   - num 须为 N2_VEC_GROUP 整倍数
__aicore__ inline void SolveAIV::ProcessN2Vec(uint64_t base, uint32_t num)
{
    auto eventId = EVENT_ID0;
    constexpr uint32_t G = N2_VEC_GROUP;
    const uint32_t nCols = this->cols;
    const uint32_t twoCols = nCols * 2U;
    LocalTensor<float> raw = uBuf.Get<float>();
    // 布局: aStg(256) | bStg(64*2cols) | xStg(2*G) | 25 个工作向量(64) | 索引表
    LocalTensor<float> aStg = raw[0];
    LocalTensor<float> bStg = raw[G * 4];
    LocalTensor<float> xStg = raw[G * 4 + G * twoCols];
    const uint32_t wOff = G * 4 + G * twoCols + 2 * G;
    LocalTensor<float> a00v = raw[wOff];
    LocalTensor<float> a01v = raw[wOff + G];
    LocalTensor<float> a10v = raw[wOff + G * 2];
    LocalTensor<float> a11v = raw[wOff + G * 3];
    LocalTensor<float> sq0 = raw[wOff + G * 4];
    LocalTensor<float> sq1 = raw[wOff + G * 5];
    LocalTensor<float> p0 = raw[wOff + G * 6];
    LocalTensor<float> p1 = raw[wOff + G * 7];
    LocalTensor<float> q0 = raw[wOff + G * 8];
    LocalTensor<float> q1 = raw[wOff + G * 9];
    LocalTensor<float> f = raw[wOff + G * 10];
    LocalTensor<float> fu = raw[wOff + G * 11];
    LocalTensor<float> u = raw[wOff + G * 12];
    LocalTensor<float> invU = raw[wOff + G * 13];
    LocalTensor<float> invP0 = raw[wOff + G * 14];
    LocalTensor<float> ones = raw[wOff + G * 15];
    LocalTensor<float> b0c = raw[wOff + G * 16];
    LocalTensor<float> b1c = raw[wOff + G * 17];
    LocalTensor<float> r0 = raw[wOff + G * 18];
    LocalTensor<float> r1 = raw[wOff + G * 19];
    LocalTensor<float> y1 = raw[wOff + G * 20];
    LocalTensor<float> t = raw[wOff + G * 21];
    LocalTensor<float> x1c = raw[wOff + G * 22];
    LocalTensor<float> x0c = raw[wOff + G * 23];
    LocalTensor<float> sF = raw[wOff + G * 24]; // 主元指示子: 1=下主元(换行), 0=上主元
    const uint32_t iOff = wOff + G * 25;
    LocalTensor<uint32_t> idxA0 = raw[iOff].ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> idxA1 = raw[iOff + G].ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> idxA2 = raw[iOff + G * 2].ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> idxA3 = raw[iOff + G * 3].ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> idxB0 = raw[iOff + G * 4].ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> idxB1 = raw[iOff + G * 5].ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> idxPair = raw[iOff + G * 6].ReinterpretCast<uint32_t>(); // 跨两槽 128 项
    // 索引表: S 管写一次, V 管读; 索引与 base 为字节单位
    for (uint32_t k = 0; k < G; k++) {
        idxA0.SetValue(k, k * 16U);
        idxA1.SetValue(k, k * 16U + 4U);
        idxA2.SetValue(k, k * 16U + 8U);
        idxA3.SetValue(k, k * 16U + 12U);
        idxB0.SetValue(k, k * twoCols * 4U);
        idxB1.SetValue(k, (k * twoCols + nCols) * 4U);
        // 交错装包索引(x0c/x1c → 相邻对)
        idxPair.SetValue(2 * k, (G + k) * 4U);
        idxPair.SetValue(2 * k + 1, k * 4U);
    }
    Duplicate<float>(ones, 1.0f, G);
    pipe_barrier(PIPE_ALL); // 索引/ones 写入完成后 V 管方可读

    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyExtParams cpA{1, G * 4U * sizeof(float), 0, 0, 0};
    DataCopyExtParams cpB{1, static_cast<uint32_t>(G * twoCols * sizeof(float)), 0, 0, 0};
    DataCopyExtParams cpX{1, 2 * G * sizeof(float), 0, 0, 0}; // 整组交错 X 一条写出

    for (uint32_t m0 = 0; m0 < num; m0 += G) {
        uint64_t mBase = base + m0;
        // ① 整组连续搬运 A/B(各 1 条 MTE, 拆包走 V 管)
        DataCopyPad(aStg, aGM[mBase * 4U], cpA, padParams);
        DataCopyPad(bStg, bGM[mBase * twoCols], cpB, padParams);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        // ② 拆包 A: 交错 [a00,a01,a10,a11]×64 → 四个 64 车道向量
        Gather<float>(a00v, aStg, idxA0, 0, G);
        Gather<float>(a01v, aStg, idxA1, 0, G);
        Gather<float>(a10v, aStg, idxA2, 0, G);
        Gather<float>(a11v, aStg, idxA3, 0, G);
        pipe_barrier(PIPE_V);

        // ③ d=a00²−a10², s=(d<0), om=1−s; 乘 0/1 精确混合(避免双舍入放大病态误差)
        Mul<float>(sq0, a00v, a00v, G, 1, {1, 1, 1, 8, 8, 8});
        Mul<float>(sq1, a10v, a10v, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Sub<float>(y1, sq0, sq1, G, 1, {1, 1, 1, 8, 8, 8}); // d
        pipe_barrier(PIPE_ALL);                             // V 写 d → S 标量读
        for (uint32_t k = 0; k < G; k++) {
            sF.SetValue(k, y1.GetValue(k) < 0.0f ? 1.0f : 0.0f);
        }
        pipe_barrier(PIPE_ALL);                              // S 写 sF → V 读
        Sub<float>(sq0, ones, sF, G, 1, {1, 1, 1, 8, 8, 8}); // om = 1−s(复用 sq0 槽)
        pipe_barrier(PIPE_V);
        Mul<float>(y1, a00v, sq0, G, 1, {1, 1, 1, 8, 8, 8});
        Mul<float>(t, a10v, sF, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Add<float>(p0, y1, t, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Mul<float>(y1, a01v, sq0, G, 1, {1, 1, 1, 8, 8, 8});
        Mul<float>(t, a11v, sF, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Add<float>(p1, y1, t, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Mul<float>(y1, a10v, sq0, G, 1, {1, 1, 1, 8, 8, 8});
        Mul<float>(t, a00v, sF, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Add<float>(q0, y1, t, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Mul<float>(y1, a11v, sq0, G, 1, {1, 1, 1, 8, 8, 8});
        Mul<float>(t, a01v, sF, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Add<float>(q1, y1, t, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Div<float>(invP0, ones, p0, G, 1, {1, 1, 1, 8, 8, 8}); // ia = 1/p0(与标量路径一致)
        pipe_barrier(PIPE_V);
        Mul<float>(f, q0, invP0, G, 1, {1, 1, 1, 8, 8, 8}); // f = q0·ia(与标量路径同舍入)
        pipe_barrier(PIPE_V);
        Mul<float>(fu, f, p1, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Sub<float>(u, q1, fu, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);
        Div<float>(invU, ones, u, G, 1, {1, 1, 1, 8, 8, 8});
        pipe_barrier(PIPE_V);

        // ④ 每列: 拆 b 行 → 混合角色 → 单套公式 → 写出(2D 跨距写仅首块生效, 逐矩阵拷贝)
        for (uint32_t c = 0; c < nCols; c++) {
            Gather<float>(b0c, bStg, idxB0, c * sizeof(float), G);
            Gather<float>(b1c, bStg, idxB1, c * sizeof(float), G);
            pipe_barrier(PIPE_V);
            Mul<float>(y1, b0c, sq0, G, 1, {1, 1, 1, 8, 8, 8}); // b0·om
            Mul<float>(t, b1c, sF, G, 1, {1, 1, 1, 8, 8, 8});   // b1·s
            pipe_barrier(PIPE_V);
            Add<float>(r0, y1, t, G, 1, {1, 1, 1, 8, 8, 8}); // 精确: s=0→b0, s=1→b1
            pipe_barrier(PIPE_V);
            Mul<float>(y1, b1c, sq0, G, 1, {1, 1, 1, 8, 8, 8}); // b1·om
            Mul<float>(t, b0c, sF, G, 1, {1, 1, 1, 8, 8, 8});   // b0·s
            pipe_barrier(PIPE_V);
            Add<float>(r1, y1, t, G, 1, {1, 1, 1, 8, 8, 8}); // s=0→b1, s=1→b0
            pipe_barrier(PIPE_V);
            Mul<float>(fu, f, r0, G, 1, {1, 1, 1, 8, 8, 8});
            pipe_barrier(PIPE_V);
            Sub<float>(y1, r1, fu, G, 1, {1, 1, 1, 8, 8, 8}); // y1 = R1 − f·R0
            pipe_barrier(PIPE_V);
            Mul<float>(x1c, y1, invU, G, 1, {1, 1, 1, 8, 8, 8});
            pipe_barrier(PIPE_V);
            Mul<float>(t, p1, x1c, G, 1, {1, 1, 1, 8, 8, 8});
            pipe_barrier(PIPE_V);
            Sub<float>(x0c, r0, t, G, 1, {1, 1, 1, 8, 8, 8});
            pipe_barrier(PIPE_V);
            Mul<float>(x0c, x0c, invP0, G, 1, {1, 1, 1, 8, 8, 8});
            pipe_barrier(PIPE_V);
            // 交错装包后单条写出(4B 散写不可靠)
            Gather<float>(xStg, x1c, idxPair, 0, 2 * G);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            DataCopyPad(xGM[mBase * twoCols], xStg, cpX);
            SetFlag<HardEvent::MTE3_V>(eventId);
            WaitFlag<HardEvent::MTE3_V>(eventId); // 写完成才准下一列 V 复写 x0c/x1c
            // V_MTE2: Gather 读完 aStg/bStg 后才准下一组 MTE2 复写(V_MTE3 不覆盖 V→MTE2)
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
        }
    }
    pipe_barrier(PIPE_ALL);
}

// n=2 批量标量路径: chunk 整搬 A/B、标量解 2x2、整体搬出, 均摊小矩阵搬运与同步开销
// chunk 布局(从 uBuf 基址重划): [aBuf0|aBuf1|bBuf0|bBuf1|xBuf], 双缓冲预取
__aicore__ inline void SolveAIV::ProcessN2()
{
    auto eventId = EVENT_ID0;
    // chunk 由预算反推(8C+6C·cols ≤ 预算, 见 SolveN2ChunkCap), 不设人为上限
    uint32_t chunk = SolveN2ChunkCap(this->cols);
    LocalTensor<float> raw = uBuf.Get<float>();
    const uint32_t aOff[2] = {0U, 4U * chunk};
    const uint32_t bOff[2] = {8U * chunk, 8U * chunk + 2U * chunk * this->cols};
    const uint32_t xOff = bOff[1] + 2U * chunk * this->cols;

    // 小 cols 且聚合满整组: 前 vecNum 个矩阵走向量路径, 尾数回落标量路径
    uint32_t vecNum = 0;
    if (this->cols <= N2_VEC_MAX_COLS) {
        vecNum = this->batchNum / N2_VEC_GROUP * N2_VEC_GROUP;
    }
    if (vecNum > 0) {
        ProcessN2Vec(this->batchStart, vecNum);
    }

    uint64_t base = this->batchStart + vecNum;
    uint32_t remain = this->batchNum - vecNum;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if (remain == 0U) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    // 预取第 0 块
    uint32_t curCnt = remain < chunk ? remain : chunk;
    remain -= curCnt;
    {
        DataCopyExtParams cpA{1, static_cast<uint32_t>(curCnt * 4U * sizeof(float)), 0, 0, 0};
        DataCopyExtParams cpB{1, static_cast<uint32_t>(curCnt * 2U * this->cols * sizeof(float)), 0, 0, 0};
        DataCopyPad(raw[aOff[0]], aGM[base * 4U], cpA, padParams);
        DataCopyPad(raw[bOff[0]], bGM[base * 2U * this->cols], cpB, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventId);

    uint32_t cur = 0U;
    while (true) {
        WaitFlag<HardEvent::MTE2_V>(eventId);
        pipe_barrier(PIPE_ALL);

        bool hasNext = remain > 0U;
        uint32_t nxtCnt = 0U;
        if (hasNext) {
            nxtCnt = remain < chunk ? remain : chunk;
            remain -= nxtCnt;
            uint32_t nxt = cur ^ 1U;
            DataCopyExtParams cpAn{1, static_cast<uint32_t>(nxtCnt * 4U * sizeof(float)), 0, 0, 0};
            DataCopyExtParams cpBn{1, static_cast<uint32_t>(nxtCnt * 2U * this->cols * sizeof(float)), 0, 0, 0};
            DataCopyPad(raw[aOff[nxt]], aGM[(base + curCnt) * 4U], cpAn, padParams);
            DataCopyPad(raw[bOff[nxt]], bGM[(base + curCnt) * 2U * this->cols], cpBn, padParams);
            SetFlag<HardEvent::MTE2_V>(eventId);
        }

        LocalTensor<float> aBuf = raw[aOff[cur]];
        LocalTensor<float> bBuf = raw[bOff[cur]];
        LocalTensor<float> xBuf = raw[xOff];
        // 标量解 2x2(行主元): |a00|≥|a10| 取 a00 为主元, 否则行交换取 a10
        for (uint32_t m = 0; m < curCnt; m++) {
            const uint32_t ao = m * 4U;
            float a00 = aBuf.GetValue(ao + 0);
            float a01 = aBuf.GetValue(ao + 1);
            float a10 = aBuf.GetValue(ao + 2);
            float a11 = aBuf.GetValue(ao + 3);
            float aa0 = a00 < 0 ? -a00 : a00;
            float aa1 = a10 < 0 ? -a10 : a10;
            const uint32_t bo = m * 2U * this->cols;
            if (aa0 >= aa1) {
                // 以乘代除: 3 次除法 → 2 次, cols 循环内无除法
                float ia = 1.0f / a00;
                float f = a10 * ia;
                float u = a11 - f * a01;
                float iu = 1.0f / u;
                for (uint32_t col = 0; col < this->cols; col++) {
                    float b0 = bBuf.GetValue(bo + col);
                    float b1 = bBuf.GetValue(bo + this->cols + col);
                    float x1 = (b1 - f * b0) * iu;
                    float x0 = (b0 - a01 * x1) * ia;
                    xBuf.SetValue(bo + col, x0);
                    xBuf.SetValue(bo + this->cols + col, x1);
                }
            } else {
                // 行交换后主元 a10
                float ia = 1.0f / a10;
                float f = a00 * ia;
                float v = a01 - f * a11;
                float iv = 1.0f / v;
                for (uint32_t col = 0; col < this->cols; col++) {
                    float b0 = bBuf.GetValue(bo + col);
                    float b1 = bBuf.GetValue(bo + this->cols + col);
                    float x1 = (b0 - f * b1) * iv;
                    float x0 = (b1 - a11 * x1) * ia;
                    xBuf.SetValue(bo + col, x0);
                    xBuf.SetValue(bo + this->cols + col, x1);
                }
            }
        }
        pipe_barrier(PIPE_ALL);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        {
            DataCopyExtParams cpX{1, static_cast<uint32_t>(curCnt * 2U * this->cols * sizeof(float)), 0, 0, 0};
            DataCopyPad(xGM[base * 2U * this->cols], xBuf, cpX);
        }
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);

        base += curCnt;
        if (!hasNext) {
            break;
        }
        curCnt = nxtCnt;
        cur ^= 1U;
    }
    pipe_barrier(PIPE_ALL);
}

// 遍历本核任务段; g → (batchIdx, blkIdx) = (g/colBlkNum, g%colBlkNum), 同矩阵块连续共享分解
__aicore__ inline void SolveAIV::ProcessLu()
{
    auto eventId = EVENT_ID0;
    uint32_t taskEnd = this->taskFirst + this->taskCnt;
    uint32_t taskId = this->taskFirst;
    uint32_t curBatchIdx = 0xFFFFFFFFU;
    while (taskId < taskEnd) {
        uint32_t batchIdx = taskId / this->colBlkNum;
        uint32_t blkIdx = taskId % this->colBlkNum;

        if (batchIdx != curBatchIdx) {
            // 新矩阵: 装载并分解(上一矩阵的 StoreX 已由 MTE3_V wait 保证完成)
            LoadA(batchIdx);
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            DecomposeLu();

            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
            curBatchIdx = batchIdx;
        }

        uint32_t blkCols = this->cols - blkIdx * this->colsPerBlk; // 本块真实列数(尾块可不满)
        if (blkCols > this->colsPerBlk) {
            blkCols = this->colsPerBlk;
        }
        uint32_t vecWidth = (blkCols + REPEAT_ELEMS - 1) / REPEAT_ELEMS * REPEAT_ELEMS; // 64 对齐的指令宽度

        LoadBPerm(batchIdx, blkIdx, blkCols); // bBlk ← P·B 块(按 perm 重排)
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        FwdSub(vecWidth);  // bBlk: P·B → y
        BackSub(vecWidth); // xBlk: y → x

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        StoreX(batchIdx, blkIdx, blkCols);

        // 等本块 StoreX(MTE3 读 xBlk) 完成, 再由 V 写下一块 xBlk
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        taskId++;
    }
    pipe_barrier(PIPE_ALL);
}

// ws 副本行段交换(分块 LU 主元)
__aicore__ inline void SolveAIV::GMSwapSeg(uint64_t wsBase, uint32_t rowA, uint32_t rowB, uint32_t c0, uint32_t c1,
                                           LocalTensor<float> t1, LocalTensor<float> t2)
{
    if (c1 <= c0) {
        return;
    }
    uint32_t segLen = c1 - c0;
    DataCopyExtParams cp{1, static_cast<uint32_t>(segLen * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(t1, wsGM[wsBase + rowA * this->n + c0], cp, pad);
    DataCopyPad(t2, wsGM[wsBase + rowB * this->n + c0], cp, pad);
    // SetFlag/WaitFlag 不阻塞标量; 标量观测 MTE2 数据须 PIPE_ALL
    pipe_barrier(PIPE_ALL);
    DataCopyPad(wsGM[wsBase + rowA * this->n + c0], t2, cp);
    DataCopyPad(wsGM[wsBase + rowB * this->n + c0], t1, cp);
    pipe_barrier(PIPE_ALL); // 标量等 MTE3 写回完(t1/t2 复用与 ws 读序)
}

// 分块 LU(n>64): A 驻留 GM ws 副本(in-place 变 L/U), UB 每次一个 64 列 panel + 右侧条;
// 分解按 panel 循环(主元含 GM 行段交换), 求解系数按需从 GM 搬; UB 两阶段重划, perm 固定尾部
__aicore__ inline void SolveAIV::ProcessLuBig()
{
    auto eventId = EVENT_ID0;
    const uint32_t nDim = this->n;
    const uint32_t cpb = this->colsPerBlk;
    LocalTensor<float> raw = uBuf.Get<float>();
    const uint32_t PERM_OFF = MAX_N * ROW_STRIDE * 2 + ROW_STRIDE + MAX_COLS_PER_BLK * 2 + MAX_B_BLK_ELEMS * 2 -
                              BIG_N; // 44864: 尾部 256, 两阶段布局均不覆盖
    LocalTensor<float> permBuf = raw[PERM_OFF];

    uint32_t taskEnd = this->taskFirst + this->taskCnt;
    uint32_t taskId = this->taskFirst;
    uint32_t curBatchIdx = 0xFFFFFFFFU;
    uint64_t wsBase = 0; // 当前矩阵在 ws 副本的基址
    while (taskId < taskEnd) {
        uint32_t batchIdx = taskId / this->colBlkNum;
        uint32_t blkIdx = taskId % this->colBlkNum;

        if (batchIdx != curBatchIdx) {
            curBatchIdx = batchIdx;
            wsBase = static_cast<uint64_t>(batchIdx) * nDim * nDim;
            // ---- 分解阶段布局 ----
            //   pnl[BIG_N][PANEL_W] 16384 | swapT1/2[BIG_N] 512 | strip[BIG_N][PANEL_W] 16384
            //   bcast/mulOut/luScratch/srcRow[PANEL_W] 256
            //   strip = 右侧整条(行[k,n)×64列), 顶部行(U12)与下方行(T)单趟合并处理, 见右侧更新
            LocalTensor<float> pnl = raw[0];
            LocalTensor<float> swapT1 = raw[BIG_N * PANEL_W];
            LocalTensor<float> swapT2 = raw[BIG_N * PANEL_W + BIG_N];
            LocalTensor<float> strip = raw[BIG_N * PANEL_W + BIG_N * 2];
            LocalTensor<float> bc = raw[BIG_N * PANEL_W * 2 + BIG_N * 2];
            LocalTensor<float> mo = raw[BIG_N * PANEL_W * 2 + BIG_N * 2 + PANEL_W];
            LocalTensor<float> luScratch = mo[PANEL_W];     // 消元行临时(64)
            LocalTensor<float> srcRow = luScratch[PANEL_W]; // 消元源行(L 区清零后)

            // A 拷入 ws 副本(n×n 超 UB, 分条 GM↔UB↔GM 中转)
            {
                uint32_t total = nDim * nDim;
                uint32_t seg = BIG_N * PANEL_W; // 16384 float/条
                DataCopyPadExtParams<float> padAll{false, 0, 0, 0};
                for (uint32_t off = 0; off < total; off += seg) {
                    uint32_t len = (total - off < seg) ? (total - off) : seg;
                    DataCopyExtParams cpSeg{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
                    DataCopyPad(raw[0], aGM[static_cast<uint64_t>(batchIdx) * nDim * nDim + off], cpSeg, padAll);
                    pipe_barrier(PIPE_ALL); // 标量等待 MTE2 完成后再发 MTE3(UB→GM 中转)
                    DataCopyPad(wsGM[wsBase + off], raw[0], cpSeg);
                    pipe_barrier(PIPE_ALL); // 标量等 MTE3 完成后下一段才可复用 raw
                }
            }

            for (uint32_t r = 0; r < nDim; r++) {
                permBuf.SetValue(r, static_cast<float>(static_cast<int32_t>(r)));
            }
            pipe_barrier(PIPE_ALL);

            // ---- panel 循环 ----
            for (uint32_t k = 0; k < nDim; k += PANEL_W) {
                uint32_t pw = (nDim - k < PANEL_W) ? (nDim - k) : PANEL_W; // panel 实际列宽
                uint32_t rows = nDim - k;                                  // panel 行数(含下方 L21)
                // 搬 panel: pnl[i] ← ws[k+i][k..k+pw)
                DataCopyPadExtParams<float> padP{false, 0, 0, 0};
                for (uint32_t i = 0; i < rows; i++) {
                    DataCopyExtParams cpP{1, static_cast<uint32_t>(pw * sizeof(float)), 0, 0, 0};
                    DataCopyPad(pnl[i * PANEL_W], wsGM[wsBase + (k + i) * nDim + k], cpP, padP);
                }
                SetFlag<HardEvent::MTE2_V>(eventId);
                WaitFlag<HardEvent::MTE2_V>(eventId);

                // panel 内逐列分解(与整装 DecomposeLu 同构; 主元交换需同步换 GM 段与 perm)
                for (uint32_t jc = 0; jc < pw; jc++) {
                    pipe_barrier(PIPE_ALL);
                    uint32_t pivot = jc;
                    float bestAbs = pnl.GetValue(jc * PANEL_W + jc);
                    bestAbs = bestAbs < 0 ? -bestAbs : bestAbs;
                    for (uint32_t r = jc + 1; r < rows; r++) {
                        float v = pnl.GetValue(r * PANEL_W + jc);
                        v = v < 0 ? -v : v;
                        if (v > bestAbs) {
                            bestAbs = v;
                            pivot = r;
                        }
                    }
                    if (pivot != jc) {
                        for (uint32_t c = 0; c < PANEL_W; c++) { // pnl 整行交换(64 宽)
                            float t = pnl.GetValue(jc * PANEL_W + c);
                            pnl.SetValue(jc * PANEL_W + c, pnl.GetValue(pivot * PANEL_W + c));
                            pnl.SetValue(pivot * PANEL_W + c, t);
                        }
                        pipe_barrier(PIPE_ALL);
                        // GM 段交换: 已完成 L 区 [0,k) + 未处理区 [k+pw,n)
                        GMSwapSeg(wsBase, k + jc, k + pivot, 0, k, swapT1, swapT2);
                        GMSwapSeg(wsBase, k + jc, k + pivot, k + pw, nDim, swapT1, swapT2);
                        float tp = permBuf.GetValue(k + jc);
                        permBuf.SetValue(k + jc, permBuf.GetValue(k + pivot));
                        permBuf.SetValue(k + pivot, tp);
                        pipe_barrier(PIPE_ALL);
                    }
                    float piv = pnl.GetValue(jc * PANEL_W + jc);
                    float ipiv = 1.0f / piv;
                    // 构造消元源行: [0,jc] 清零, (jc,PANEL_W) 拷主元行
                    // (pnl 就地存 L, 整装路径"下三角恒 0"前提在此不成立, 全宽减会污染 L 乘子)
                    for (uint32_t c = 0; c <= jc; c++) {
                        srcRow.SetValue(c, 0.0f);
                    }
                    for (uint32_t c = jc + 1; c < PANEL_W; c++) {
                        srcRow.SetValue(c, pnl.GetValue(jc * PANEL_W + c));
                    }
                    pipe_barrier(PIPE_ALL);
                    for (uint32_t r = jc + 1; r < rows; r++) {
                        float lFactor = pnl.GetValue(r * PANEL_W + jc) * ipiv;
                        pipe_barrier(PIPE_ALL);
                        Duplicate<float>(bc, lFactor, PANEL_W);
                        pipe_barrier(PIPE_V);
                        Mul(luScratch, bc, srcRow, REPEAT_ELEMS, 1, {1, 1, 1, 8, 8, 8});
                        pipe_barrier(PIPE_V);
                        Sub(pnl[r * PANEL_W], pnl[r * PANEL_W], luScratch, REPEAT_ELEMS, 1, {1, 1, 1, 8, 8, 8});
                        pipe_barrier(PIPE_V);
                    }
                    // L 补写: 全宽 Sub 会覆盖行内标量写(matL 独立缓冲无此问题), 须等 V 排空后重算
                    pipe_barrier(PIPE_ALL);
                    for (uint32_t r = jc + 1; r < rows; r++) {
                        pnl.SetValue(r * PANEL_W + jc, pnl.GetValue(r * PANEL_W + jc) * ipiv);
                    }
                    pipe_barrier(PIPE_ALL);
                }
                pipe_barrier(PIPE_ALL);

                // 右侧更新: 列 [k+pw, n) 按 64 列条(整条驻留 strip, 行[k,n)×64列)
                for (uint32_t c = k + pw; c < nDim; c += PANEL_W) {
                    uint32_t cb = (nDim - c < PANEL_W) ? (nDim - c) : PANEL_W;
                    uint32_t vw = (cb + REPEAT_ELEMS - 1) / REPEAT_ELEMS * REPEAT_ELEMS;
                    uint8_t rep = static_cast<uint8_t>(vw / REPEAT_ELEMS);
                    // 搬整条: strip[i] ← ws[k+i][c..c+cb), i ∈ [0, rows)
                    DataCopyPadExtParams<float> padC{false, 0, 0, 0};
                    for (uint32_t i = 0; i < rows; i++) {
                        DataCopyExtParams cpS{1, static_cast<uint32_t>(cb * sizeof(float)), 0, 0, 0};
                        DataCopyPad(strip[i * PANEL_W], wsGM[wsBase + (k + i) * nDim + c], cpS, padC);
                    }
                    SetFlag<HardEvent::MTE2_V>(eventId);
                    WaitFlag<HardEvent::MTE2_V>(eventId);

                    // 单趟重放(乘子均为最终值): 顶部行做 U12 前代, 下方行做 T −= L21·U12,
                    //   系数统一从 pnl 取, 免双循环
                    pipe_barrier(PIPE_ALL);
                    for (uint32_t i = 1; i < rows; i++) {
                        uint32_t jEnd = (i < pw) ? i : pw;
                        for (uint32_t j = 0; j < jEnd; j++) {
                            float lCoef = pnl.GetValue(i * PANEL_W + j);
                            pipe_barrier(PIPE_ALL);
                            Axpy(strip[i * PANEL_W], strip[j * PANEL_W], -lCoef, REPEAT_ELEMS, rep, {1, 1, 8, 8});
                            pipe_barrier(PIPE_V);
                        }
                    }
                    pipe_barrier(PIPE_ALL);

                    // 写回整条: ws[k+i][c..c+cb) ← strip[i]
                    SetFlag<HardEvent::V_MTE3>(eventId);
                    WaitFlag<HardEvent::V_MTE3>(eventId);
                    for (uint32_t i = 0; i < rows; i++) {
                        DataCopyExtParams cpS{1, static_cast<uint32_t>(cb * sizeof(float)), 0, 0, 0};
                        DataCopyPad(wsGM[wsBase + (k + i) * nDim + c], strip[i * PANEL_W], cpS);
                    }
                    pipe_barrier(PIPE_ALL); // MTE3 写回完成后下一列条方可复用 strip
                }

                // 写回 panel(含 L 区与 U 区)
                SetFlag<HardEvent::V_MTE3>(eventId);
                WaitFlag<HardEvent::V_MTE3>(eventId);
                DataCopyPadExtParams<float> padW{false, 0, 0, 0};
                for (uint32_t i = 0; i < rows; i++) {
                    DataCopyExtParams cpW{1, static_cast<uint32_t>(pw * sizeof(float)), 0, 0, 0};
                    DataCopyPad(wsGM[wsBase + (k + i) * nDim + k], pnl[i * PANEL_W], cpW);
                }
                pipe_barrier(PIPE_ALL); // MTE3 写回完成后下一 panel 的 MTE2 方可读 ws
            }
            // 分解完成: ws = L(严格下三角) + U(上三角), permBuf 就绪
        }

        // ---- 求解阶段布局 ----(perm 仍在尾部 PERM_OFF)
        //   bBlk ≤16384 @0 | xBlk ≤16384 @16384 | bc[vw≤192] @32768
        //   panelBuf[PBLK_W²] @32960 | tileBuf[TILE_R×PBLK_W] @35264
        LocalTensor<float> bBlk = raw[0];
        LocalTensor<float> xBlk = raw[BIG_N * PANEL_W];
        LocalTensor<float> bc = raw[BIG_N * PANEL_W * 2];
        LocalTensor<float> panelBuf = raw[BIG_N * PANEL_W * 2 + 192];
        LocalTensor<float> tileBuf = raw[BIG_N * PANEL_W * 2 + 192 + SOLVE_PBLK_W * SOLVE_PBLK_W];

        uint32_t blkCols = this->cols - blkIdx * cpb;
        if (blkCols > cpb) {
            blkCols = cpb;
        }
        uint32_t vw = (blkCols + REPEAT_ELEMS - 1) / REPEAT_ELEMS * REPEAT_ELEMS;
        uint8_t rep = static_cast<uint8_t>(vw / REPEAT_ELEMS);

        DataCopyPadExtParams<float> padB{false, 0, 0, 0};
        for (uint32_t r = 0; r < nDim; r++) {
            uint32_t src = static_cast<uint32_t>(static_cast<int32_t>(permBuf.GetValue(r)));
            DataCopyExtParams cpB{1, static_cast<uint32_t>(blkCols * sizeof(float)), 0, 0, 0};
            DataCopyPad(bBlk[r * cpb], bGM[(static_cast<uint64_t>(batchIdx) * nDim + src) * this->cols + blkIdx * cpb],
                        cpB, padB);
        }
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        // 前代: L·y=P·B, 面板分块(TailUpdate 式): [面板块 → 面板内解 → 尾端条更新],
        // 累加顺序与逐行版逐位一致; 系数块复写前显式 V_MTE2(PIPE_V 不保证 V→MTE2 隔离)
        for (uint32_t p0 = 0; p0 < nDim; p0 += SOLVE_PBLK_W) {
            uint32_t pw2 = (nDim - p0 < SOLVE_PBLK_W) ? nDim - p0 : SOLVE_PBLK_W;
            // a) L 面板块 [p0, p0+pw2)² 逐行发出, 一次同步
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
            for (uint32_t i = 0; i < pw2; i++) {
                DataCopyExtParams cpP{1, static_cast<uint32_t>(pw2 * sizeof(float)), 0, 0, 0};
                DataCopyPad(panelBuf[i * SOLVE_PBLK_W], wsGM[wsBase + (p0 + i) * nDim + p0], cpP, padB);
            }
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            pipe_barrier(PIPE_ALL);
            // b) 面板内顺序前代(乘数 bBlk[p0+j] 已被先前面板的尾端更新完毕)
            for (uint32_t i = 0; i < pw2; i++) {
                for (uint32_t j = 0; j < i; j++) {
                    float lCoef = panelBuf.GetValue(i * SOLVE_PBLK_W + j);
                    pipe_barrier(PIPE_ALL);
                    Axpy(bBlk[(p0 + i) * cpb], bBlk[(p0 + j) * cpb], -lCoef, REPEAT_ELEMS, rep, {1, 1, 8, 8});
                    pipe_barrier(PIPE_V);
                }
            }
            // c) 尾端: 行 [p0+pw2, n) 按 TILE_R 行条批量搬系数再逐行逐列更新
            for (uint32_t t0 = p0 + pw2; t0 < nDim; t0 += SOLVE_TILE_R) {
                uint32_t rt = (nDim - t0 < SOLVE_TILE_R) ? nDim - t0 : SOLVE_TILE_R;
                SetFlag<HardEvent::V_MTE2>(eventId);
                WaitFlag<HardEvent::V_MTE2>(eventId);
                for (uint32_t i = 0; i < rt; i++) {
                    DataCopyExtParams cpT{1, static_cast<uint32_t>(pw2 * sizeof(float)), 0, 0, 0};
                    DataCopyPad(tileBuf[i * SOLVE_PBLK_W], wsGM[wsBase + (t0 + i) * nDim + p0], cpT, padB);
                }
                SetFlag<HardEvent::MTE2_V>(eventId);
                WaitFlag<HardEvent::MTE2_V>(eventId);
                pipe_barrier(PIPE_ALL);
                for (uint32_t i = 0; i < rt; i++) {
                    for (uint32_t j = 0; j < pw2; j++) {
                        float lCoef = tileBuf.GetValue(i * SOLVE_PBLK_W + j);
                        pipe_barrier(PIPE_ALL);
                        Axpy(bBlk[(t0 + i) * cpb], bBlk[(p0 + j) * cpb], -lCoef, REPEAT_ELEMS, rep, {1, 1, 8, 8});
                        pipe_barrier(PIPE_V);
                    }
                }
            }
        }
        // 回代: U·x=y, 面板自底向上: [U 面板块 → 面板内解 → 上方行尾端更新];
        // 累加顺序与逐行版不同, 均为合法回代序(同精度级)
        for (int32_t pp = static_cast<int32_t>((nDim - 1) / SOLVE_PBLK_W); pp >= 0; pp--) {
            uint32_t p0 = static_cast<uint32_t>(pp) * SOLVE_PBLK_W;
            uint32_t pw2 = (nDim - p0 < SOLVE_PBLK_W) ? nDim - p0 : SOLVE_PBLK_W;
            // a) U 面板块 [p0, p0+pw2)² (行 i 的 [p0, p0+pw2) 段, 含对角)
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
            for (uint32_t i = 0; i < pw2; i++) {
                DataCopyExtParams cpP{1, static_cast<uint32_t>(pw2 * sizeof(float)), 0, 0, 0};
                DataCopyPad(panelBuf[i * SOLVE_PBLK_W], wsGM[wsBase + (p0 + i) * nDim + p0], cpP, padB);
            }
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            pipe_barrier(PIPE_ALL);
            // b) 面板内自底向上: 减 j∈(i, pw2)(乘 xBlk 面板行, 本面板内已定) → 对角缩放进 xBlk
            for (int32_t i = static_cast<int32_t>(pw2) - 1; i >= 0; i--) {
                uint32_t ir = static_cast<uint32_t>(i);
                for (uint32_t j = ir + 1; j < pw2; j++) {
                    float uCoef = panelBuf.GetValue(ir * SOLVE_PBLK_W + j);
                    pipe_barrier(PIPE_ALL);
                    Axpy(bBlk[(p0 + ir) * cpb], xBlk[(p0 + j) * cpb], -uCoef, REPEAT_ELEMS, rep, {1, 1, 8, 8});
                    pipe_barrier(PIPE_V);
                }
                float ipiv = 1.0f / panelBuf.GetValue(ir * SOLVE_PBLK_W + ir);
                pipe_barrier(PIPE_ALL);
                Duplicate<float>(bc, ipiv, vw);
                pipe_barrier(PIPE_V);
                Mul(xBlk[(p0 + ir) * cpb], bc, bBlk[(p0 + ir) * cpb], REPEAT_ELEMS, rep, {1, 1, 1, 8, 8, 8});
                pipe_barrier(PIPE_V);
            }
            // c) 尾端(上方行 [0,p0)): 条批量搬 U 后逐行逐列减
            int32_t tStart = ((static_cast<int32_t>(p0) - 1) / static_cast<int32_t>(SOLVE_TILE_R)) *
                             static_cast<int32_t>(SOLVE_TILE_R);
            if (tStart < 0) {
                tStart = 0; // p0=0(首面板): rt=0 空转
            }
            for (int32_t t0 = tStart; t0 >= 0; t0 -= static_cast<int32_t>(SOLVE_TILE_R)) {
                uint32_t tr0 = static_cast<uint32_t>(t0);
                uint32_t rt = (tr0 + SOLVE_TILE_R > p0) ? p0 - tr0 : SOLVE_TILE_R;
                SetFlag<HardEvent::V_MTE2>(eventId);
                WaitFlag<HardEvent::V_MTE2>(eventId);
                for (uint32_t i = 0; i < rt; i++) {
                    DataCopyExtParams cpT{1, static_cast<uint32_t>(pw2 * sizeof(float)), 0, 0, 0};
                    DataCopyPad(tileBuf[i * SOLVE_PBLK_W], wsGM[wsBase + (tr0 + i) * nDim + p0], cpT, padB);
                }
                SetFlag<HardEvent::MTE2_V>(eventId);
                WaitFlag<HardEvent::MTE2_V>(eventId);
                pipe_barrier(PIPE_ALL);
                for (uint32_t i = 0; i < rt; i++) {
                    for (uint32_t j = 0; j < pw2; j++) {
                        float uCoef = tileBuf.GetValue(i * SOLVE_PBLK_W + j);
                        pipe_barrier(PIPE_ALL);
                        Axpy(bBlk[(tr0 + i) * cpb], xBlk[(p0 + j) * cpb], -uCoef, REPEAT_ELEMS, rep, {1, 1, 8, 8});
                        pipe_barrier(PIPE_V);
                    }
                }
            }
        }
        pipe_barrier(PIPE_ALL);

        // StoreX
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        for (uint32_t j = 0; j < nDim; j++) {
            DataCopyExtParams cpX{1, static_cast<uint32_t>(blkCols * sizeof(float)), 0, 0, 0};
            DataCopyPad(xGM[(static_cast<uint64_t>(batchIdx) * nDim + j) * this->cols + blkIdx * cpb], xBlk[j * cpb],
                        cpX);
        }
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        taskId++;
    }
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void SolveAIV::Process()
{
    if (this->useN2) {
        ProcessN2();
        return;
    }
    if (this->n <= MAX_N) {
        ProcessLu();
        return;
    }
    ProcessLuBig();
}

} // namespace SolveOp

#endif // ASDSIP_SOLVE_AIV_H
