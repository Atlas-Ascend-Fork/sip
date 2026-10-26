/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * P2 每核独立路径 —— 纯 AIV kernel (vector channel):
 *
 * 背景: MIX 版 sgetri_batched 的 LU 面板分解在 blockM<=512 分支只由
 * core0 执行 (lu_custom.hpp L120-124), 39/40 核全程空等; 批内矩阵串行。
 * P2 改为纯 AIV kernel (blockDim=vecCoreNum=40, 仿 smatinv_batched):
 * 每个 AIV 核独立认领矩阵序列 (bid = blockIdx, bid += blockNum),
 * 单核完成 eye->pad->LU(面板+行交换+U12+Schur)->下三角求解->上三角
 * 求解->restore 全流程, 零跨核同步。
 *
 * 为什么必须是独立 vector kernel 而非 MIX 分支 (二分实验结论):
 *  - MIX 布局下 AIC (cube) 侧不能在 kernel 入口提前 return —— 轮0
 *    实验 (两侧 dispatch 即 return) 仍 aicore timeout, 证明 kernel
 *    结束协议由 AIC 主导, AIC 不参与即挂;
 *  - AIC 侧走旧主路径会在第一个 CrossCoreWaitFlag(0x0/0x3) 死等
 *    (P2 AIV 不再按主路径节奏 Set 这些 flag);
 *  - pad.hpp 的 rebuild_eye_real/pad_real/restore_real 内含全核
 *    barrier (CrossCoreSet/WaitFlag 0x2, 按 GetBlockNum()<<1 分片),
 *    P2 下部分核不调用即死等 —— P2 kernel 用 getri_per_core.hpp 的
 *    单核全量版替代。
 *
 * tiling: blockDim = vecCoreNum (40); kernel 侧 GetBlockNum()=40。
 * 输入/输出布局与 MIX 版完全一致 (dtype==2 时 host 无需区分)。
 */

#include "../../sgetri_batched/kernel/getri_per_core.hpp"

extern "C" __global__ __aicore__ void sgetri_p2_batched(GM_ADDR sync, GM_ADDR A_org, GM_ADDR W, GM_ADDR gather1_gm,
                                                        GM_ADDR gather2_gm, GM_ADDR gather3_gm, GM_ADDR eye_gm,
                                                        GM_ADDR A_work, GM_ADDR work_gm, GM_ADDR A_inv,
                                                        GM_ADDR workspace, GM_ADDR tiling_gm)
{
    SetSyncBaseAddr((unsigned long)sync);

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t*>(tiling_gm);
    int64_t dtype = static_cast<int64_t>(*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf));
    int64_t n = static_cast<int64_t>(*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + sizeof(uint32_t)));
    int64_t batchSize = static_cast<int64_t>(*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 2 * sizeof(uint32_t)));
    int64_t blockM = static_cast<int64_t>(*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 3 * sizeof(uint32_t)));
    int64_t blockN = static_cast<int64_t>(*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 4 * sizeof(uint32_t)));
    int64_t tileM = static_cast<int64_t>(*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 5 * sizeof(uint32_t)));

    (void)dtype; // 本 kernel 只服务 dtype==2 (host GetBestKernel 保证)
    int64_t orgM = n;
    int64_t orgN = n;

    int64_t minMN = orgM > orgN ? orgN : orgM;
    int64_t wEleNum = (minMN + blockN - 1) / blockN * blockN;
    int64_t eyeMatEleNum = ((orgN + 127) / 128 * 128) * ((orgM + 15) / 16 * 16 + 128) * 3;
    int64_t batchStride = orgM * orgN * sizeof(float);

    int aivNum = GetBlockNum();
    int aivIdx = GetBlockIdx();

    int64_t M16 = (orgM + 15) / 16 * 16;
    int64_t sN = (orgN + 127) / 128 * 128;
    int64_t aWorkSliceEle = sN * (M16 + 128) * 3; // 与 host eye 同构三段 (含 slack)
    int64_t workMEle = (M16 + 511) / 512 * 512;
    int64_t workGmSliceEle = workMEle * 16 * 2;
    // 切片数与 host BlasCgetriBatchedPlan::GenWorkAclTensor 同源:
    // min(batchSize, blockDim), blockDim 即 P2 tiling 下发的 vector 核数
    int64_t slices = batchSize < aivNum ? batchSize : aivNum;
    if (slices < 1) {
        slices = 1;
    }
    GM_ADDR my_A_work = A_work + (aivIdx % slices) * aWorkSliceEle * sizeof(float);
    GM_ADDR my_work_gm = work_gm + (aivIdx % slices) * workGmSliceEle * sizeof(float);

    for (int64_t bid = aivIdx; bid < batchSize; bid += aivNum) {
        GM_ADDR cur_W = W + bid * wEleNum * sizeof(uint32_t);
        GM_ADDR cur_eye_gm = eye_gm + bid * eyeMatEleNum * sizeof(float);
        GM_ADDR cur_A_org = A_org + bid * batchStride;
        GM_ADDR cur_A_inv = A_inv + bid * batchStride;
        custom_getri_per_core(orgM, orgN, (int)blockM, (int)blockN, (int)tileM, cur_A_org, cur_A_inv, my_A_work, cur_W,
                              my_work_gm, gather1_gm, gather2_gm, cur_eye_gm, workspace);
    }
}
