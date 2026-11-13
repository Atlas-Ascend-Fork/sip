/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_KAISER_WINDOW_KERNEL_COMMON_H
#define ASDSIP_KAISER_WINDOW_KERNEL_COMMON_H
#include <type_traits>
#include "kernel_operator.h"
#include "../tiling/tiling_data.h"

namespace KaiserWindow {
using namespace AscendC;
constexpr uint32_t TILE_ELEMENTS = 4096;
constexpr uint32_t SCRATCH_ARRAYS = 6;

__aicore__ inline AsdSip::KaiserWindowTilingData LoadTiling(GM_ADDR tiling)
{
    AsdSip::KaiserWindowTilingData data;
    auto src = reinterpret_cast<__gm__ uint32_t*>(tiling);
    data.length = src[0];
    data.denominator = src[1];
    data.perCore = src[2];
    data.dtype = src[3];
    data.beta = reinterpret_cast<__gm__ float*>(tiling)[4];
    data.inverseI0e = reinterpret_cast<__gm__ float*>(tiling)[5];
    data.inverseN = reinterpret_cast<__gm__ float*>(tiling)[6];
    data.mode = src[7];
    return data;
}

template <typename T, typename Compute>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR output, const AsdSip::KaiserWindowTilingData& data)
    {
        data_ = data;
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        pipe_.InitBuffer(queue_, 2, TILE_ELEMENTS * sizeof(T));
        pipe_.InitBuffer(scratch_, SCRATCH_ARRAYS * TILE_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(mask_, TILE_ELEMENTS / 8);
    }

    __aicore__ inline void Process()
    {
        const uint32_t begin = GetBlockIdx() * data_.perCore;
        const uint32_t end = begin + data_.perCore < data_.length ? begin + data_.perCore : data_.length;
        for (uint32_t offset = begin; offset < end; offset += TILE_ELEMENTS) {
            const uint32_t count = end - offset < TILE_ELEMENTS ? end - offset : TILE_ELEMENTS;
            auto scratch = scratch_.Get<float>();
            auto result = scratch[4 * TILE_ELEMENTS];
            if (data_.mode != 0) {
                union {
                    uint32_t bits;
                    float value;
                } nanValue;
                nanValue.bits = 0x7fc00000U;
                Duplicate(result, data_.mode == 1 ? 1.0F : nanValue.value, count);
            } else {
                Compute::Run(scratch, mask_.Get<uint8_t>(), offset, count, data_);
                const uint32_t center = data_.denominator / 2;
                if ((data_.denominator % 2 == 0) && center >= offset && center < offset + count) {
                    // I0(beta) / I0(beta) is exactly one. Restore that value
                    // after approximate vector reciprocal / transcendental math.
                    auto vectorToScalar = pipe_.FetchEventID(HardEvent::V_S);
                    SetFlag<HardEvent::V_S>(vectorToScalar);
                    WaitFlag<HardEvent::V_S>(vectorToScalar);
                    result.SetValue(center - offset, 1.0F);
                    auto scalarToVector = pipe_.FetchEventID(HardEvent::S_V);
                    SetFlag<HardEvent::S_V>(scalarToVector);
                    WaitFlag<HardEvent::S_V>(scalarToVector);
                }
            }
            auto output = queue_.AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<T, float>::value) {
                Adds(output, result, 0.0F, count);
            } else {
                Cast(output, result, RoundMode::CAST_RINT, count);
            }
            queue_.EnQue(output);
            output = queue_.DeQue<T>();
            const DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(output_[offset], output, copy);
            queue_.FreeTensor(output);
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECOUT, 2> queue_;
    TBuf<TPosition::VECCALC> scratch_;
    TBuf<TPosition::VECCALC> mask_;
    GlobalTensor<T> output_;
    AsdSip::KaiserWindowTilingData data_;
};
} // namespace KaiserWindow
#endif
