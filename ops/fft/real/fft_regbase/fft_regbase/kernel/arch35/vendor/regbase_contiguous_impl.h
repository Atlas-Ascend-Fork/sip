/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

// A full 64-complex range. The pointer need not be vector-aligned. The caller
// must prove all 128 float elements are in bounds; partial blocks use masks.
__simd_callee__ inline void LoadComplexUnaligned(__ubuf__ float* src, AscendC::Reg::RegTensor<float>& real,
                                                 AscendC::Reg::RegTensor<float>& imag)
{
    using namespace AscendC;
    Reg::RegTensor<float> first, second;
    Reg::UnalignRegForLoad alignment;
    Reg::LoadUnAlignPre(alignment, src);
    Reg::LoadUnAlign(first, alignment, src, 64);
    Reg::LoadUnAlign(second, alignment, src, 64);
    Reg::DeInterleave(real, imag, first, second);
}
