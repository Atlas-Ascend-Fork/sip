/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "solve.h"

extern "C" __global__ __aicore__ void solve(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    SolveOp::SolveAIV op;
    op.Init(a, b, x, workSpace, tilingGm);
    op.Process();
}
