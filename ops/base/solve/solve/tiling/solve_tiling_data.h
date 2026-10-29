/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_SOLVE_TILING_DATA_H
#define ASDSIP_SOLVE_TILING_DATA_H

#include <cstdint>

namespace AsdSip {
constexpr uint32_t SOLVE_MAX_CORE_CNT = 40; // 参与计算的向量核上限(910B3 实际 40)

// ---- UB 布局常量(host/kernel 共享) ----
constexpr uint32_t SOLVE_MAX_N = 256;             // 支持的最大矩阵阶数(总上限)
constexpr uint32_t SOLVE_RESIDENT_MAX_N = 64;     // 整装 LU 路径的阶数上限(A+L 可常驻 UB)
constexpr uint32_t SOLVE_ROW_STRIDE = 64;         // A 每行物理跨度(1 repeat = 64 floats)
constexpr uint32_t SOLVE_MAX_COLS_PER_BLK = 6144; // 每块的列数上限(= vecT/tmpRow 容量, 64 对齐)
constexpr uint32_t SOLVE_MAX_B_BLK_ELEMS = 12288; // B 块缓冲容量(整装路径, bRows/xRows 各 12288)
constexpr uint32_t SOLVE_BIG_B_BLK_ELEMS = 16384; // B 块缓冲容量(分块 LU 路径, n<=256 时 256×64)

// n=2 批量标量路径的 UB 总预算(与 InitBuffer 分配量一致):
//   matU+matL(2×64×64) + scratch(64) + bcastRow/mulOut(2×6144) + bBlk/xBlk(2×12288)
constexpr uint32_t SOLVE_N2_UB_BUDGET = SOLVE_RESIDENT_MAX_N * SOLVE_ROW_STRIDE * 2 + SOLVE_ROW_STRIDE +
                                        SOLVE_MAX_COLS_PER_BLK * 2 + SOLVE_MAX_B_BLK_ELEMS * 2;

// n=2 路径 chunk 容量: 布局 [aBuf0|aBuf1|bBuf0|bBuf1|xBuf], 需 8C + 6C·cols ≤ 预算
// (host/kernel 共用: ccec 端 __aicore__ 标注, g++ 端 inline)
// 结果向下取 8 的倍数: 保证 xBuf/bBuf1 基址 32B 对齐(否则 DataCopyPad 非法)
#ifdef __CCE_AICORE__
__aicore__ inline
#else
inline
#endif
    uint32_t
    SolveN2ChunkCap(uint32_t cols)
{
    // 64 位计算: 6*cols 在 uint32 域会回绕(cols 可接近 2^32), 回绕后 chunk 偏大导致 UB 越界
    const uint64_t budget = static_cast<uint64_t>(SOLVE_N2_UB_BUDGET) - 8ULL;
    const uint64_t denom = 8ULL + 6ULL * static_cast<uint64_t>(cols);
    return static_cast<uint32_t>(budget / denom) & ~7U;
}

// 走 n=2 批量标量路径的最小 chunk(矩阵数), 低于此值聚合收益不足
constexpr uint32_t SOLVE_MIN_N2_CHUNK = 8;

// tiling 布局(uint32 字):
//   [0] n            矩阵阶数
//   [1] cols         B 的总列数(右端项个数)
//   [2] colsPerBlk   每块的列数 = min(cols, 12288/n) 向下 64 对齐(受缓冲容量约束)
//   [3] colBlkNum    每矩阵的列块数 = ceil(cols / colsPerBlk)
//   [4] useN2        1=走 n=2 批量标量路径(host 判定, kernel 只读, 单一事实源)
//   [5  .. 44]  taskStart[core]   (LU 路径: 核的全局任务起点)
//   [45 .. 84]  taskNum[core]     (LU 路径: 核的任务数, 任意两核差 <= 1)
//   [85 .. 124] batchStart[core]    (N2 路径: 核的 batch 段起点)
//   [125.. 164] batchNum[core]      (N2 路径: 核的 batch 段矩阵数)
struct SolveTilingData {
    uint32_t n;
    uint32_t cols;
    uint32_t colsPerBlk;
    uint32_t colBlkNum;
    uint32_t useN2;
    uint32_t taskStart[SOLVE_MAX_CORE_CNT];
    uint32_t taskNum[SOLVE_MAX_CORE_CNT];
    uint32_t batchStart[SOLVE_MAX_CORE_CNT];
    uint32_t batchNum[SOLVE_MAX_CORE_CNT];
};
} // namespace AsdSip

#endif // ASDSIP_SOLVE_TILING_DATA_H
