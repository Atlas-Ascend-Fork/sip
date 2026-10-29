/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "solve_tiling_data.h"
#include "solve.h"
#include "solve_tiling.h"

namespace AsdSip {

AspbStatus SolveTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    void* tilingData = kernelInfo.GetTilingHostAddr();
    SolveTilingData* tilingDataPtr = reinterpret_cast<SolveTilingData*>(tilingData);
    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    OpParam::Solve param = AnyCast<OpParam::Solve>(launchParam.GetParam());
    const int64_t batch = param.batch;
    const int64_t n = param.n;
    const int64_t cols = param.nrhs;

    ASDSIP_CHECK(n >= 2 && n <= SOLVE_MAX_N, "solve: n must be in [2, 256]",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_CHECK(cols >= 1, "solve: cols(nrhs) must be >= 1", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_CHECK(batch >= 1, "solve: batch must be >= 1", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // tiling 以 uint32 存储/索引 cols(nrhs), 超出范围会被静默截断
    ASDSIP_CHECK(cols <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                 "solve: cols(nrhs) exceeds uint32 range", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    vecCoreNum = vecCoreNum > SOLVE_MAX_CORE_CNT ? SOLVE_MAX_CORE_CNT : vecCoreNum;
    vecCoreNum = (vecCoreNum == 0) ? 1 : vecCoreNum;

    // 路径判定(单一事实源): n=2 且 chunk 容量足以聚合 → 批量标量路径; 否则 LU 路径
    // (n=2 大 cols 时超出 chunk 容量, 回落至 LU 的全局任务切分)
    const bool useN2 = (n == 2) && (SolveN2ChunkCap(static_cast<uint32_t>(cols)) >= SOLVE_MIN_N2_CHUNK);

    // 列分块: n * colsPerBlk <= B 块缓冲容量; colsPerBlk 向下 64 对齐(向量 repeat 粒度), 下限 64
    // 整装路径(n<=64)容量 12288, 分块 LU 路径(n>64)容量 16384(256×64, A 不常驻腾出的空间)
    const int64_t blkCapElems = (n <= SOLVE_RESIDENT_MAX_N) ? SOLVE_MAX_B_BLK_ELEMS : SOLVE_BIG_B_BLK_ELEMS;
    int64_t blkCap = blkCapElems / n;
    int64_t colsPerBlk = (cols < blkCap) ? cols : blkCap;
    colsPerBlk = colsPerBlk / 64 * 64;
    if (colsPerBlk == 0) {
        colsPerBlk = 64;
    }
    // 每矩阵的列块数(全局任务序的步长)
    uint32_t colBlkNum = static_cast<uint32_t>((cols + colsPerBlk - 1) / colsPerBlk);
    // 任务总数以 uint32 下发到 kernel, 需保证 batch * colBlkNum 不溢出
    const uint64_t totalTask64 = static_cast<uint64_t>(batch) * colBlkNum;
    ASDSIP_CHECK(totalTask64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                 "solve: total task count exceeds uint32 range", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    for (uint32_t i = 0; i < SOLVE_MAX_CORE_CNT; i++) {
        tilingDataPtr->taskStart[i] = 0;
        tilingDataPtr->taskNum[i] = 0;
        tilingDataPtr->batchStart[i] = 0;
        tilingDataPtr->batchNum[i] = 0;
    }

    // ---- LU 路径: 全局任务 g = m*colBlkNum + b 均分(两核差 ≤1) ----
    // batch<核数时 rhs 维并行; 核内连续段共享分解, 跨核边界至多重复一次
    // 例外: n>64 时分解共享 GM ws 副本, 同矩阵列块须同核 → 按矩阵粒度切分
    if (!useN2) {
        if (n > SOLVE_RESIDENT_MAX_N) {
            for (uint32_t i = 0; i < vecCoreNum; i++) {
                // 乘法提升到 uint64, 避免 batch * i 在 uint32 域回绕
                const uint64_t first = static_cast<uint64_t>(batch) * i / vecCoreNum;
                const uint64_t last = static_cast<uint64_t>(batch) * (i + 1) / vecCoreNum;
                tilingDataPtr->taskStart[i] = static_cast<uint32_t>(first * colBlkNum);
                tilingDataPtr->taskNum[i] = static_cast<uint32_t>((last - first) * colBlkNum);
            }
        } else {
            const uint32_t totalTask = static_cast<uint32_t>(totalTask64); // 已校验不溢出
            for (uint32_t i = 0; i < vecCoreNum; i++) {
                // 乘法提升到 uint64, 避免 totalTask * i 在 uint32 域回绕
                const uint64_t first = static_cast<uint64_t>(totalTask) * i / vecCoreNum;
                const uint64_t last = static_cast<uint64_t>(totalTask) * (i + 1) / vecCoreNum;
                tilingDataPtr->taskStart[i] = static_cast<uint32_t>(first);
                tilingDataPtr->taskNum[i] = static_cast<uint32_t>(last - first);
            }
        }
    }

    // ---- N2 路径: batch 段均分(批量标量路径跨矩阵聚合, rhs 维不可切分) ----
    if (useN2) {
        uint32_t matPerCore = static_cast<uint32_t>(batch / vecCoreNum);
        uint32_t remainMatNum = static_cast<uint32_t>(batch % vecCoreNum);
        uint32_t currStart = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            uint32_t calNum = matPerCore + ((i < remainMatNum) ? 1 : 0);
            tilingDataPtr->batchStart[i] = currStart;
            tilingDataPtr->batchNum[i] = calNum;
            currStart += calNum;
        }
    }

    tilingDataPtr->n = static_cast<uint32_t>(n);
    tilingDataPtr->cols = static_cast<uint32_t>(cols);
    tilingDataPtr->colsPerBlk = static_cast<uint32_t>(colsPerBlk);
    tilingDataPtr->colBlkNum = colBlkNum;
    tilingDataPtr->useN2 = useN2 ? 1U : 0U;

    kernelInfo.SetBlockDim(vecCoreNum);
    // 分块 LU 路径(n>64)为 in-place 分解申请 A 副本(不破坏输入张量)
    size_t scratch = 0;
    if (!useN2 && n > SOLVE_RESIDENT_MAX_N) {
        scratch = static_cast<size_t>(batch) * static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    }
    kernelInfo.GetScratchSizes().push_back(scratch);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
