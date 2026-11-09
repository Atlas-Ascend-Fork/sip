/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

namespace fft_mixed {
// One 256-thread VF owns the row. For each butterfly, use its two input
// slots p and radix-p to store the sum and difference of the rotated inputs.
// This is valid for complex inputs and for odd composite groups as well.
template <bool TwoFrequencies, bool Planar = false>
__simt_vf__ __launch_bounds__(256) inline void OddSymmetricStage(__ubuf__ float* src, __ubuf__ float* dst,
                                                                 __ubuf__ float* roots, int size, int rootN, int radix,
                                                                 int prev)
{
    const int half = radix / 2;
    const int part = size / radix;
    const int outerStep = rootN / (prev * radix);
    const int innerStep = rootN / radix;
    // Planar mode writes rotated sums/differences into the other row buffer.
    // It then returns the interleaved FFT result in src. This uses the same
    // two row buffers and protects all source reads with the existing barrier.
    __ubuf__ float* values = Planar ? dst : src;
    __ubuf__ float* result = Planar ? src : dst;
    const int componentStride = Planar ? size : 1;
    const int elementStride = Planar ? 1 : 2;
    const int pairWork = (half + (Planar ? 1 : 0)) * part;
    for (int work = threadIdx.x; work < pairWork; work += 256) {
        const int p = (Planar ? 0 : 1) + work / part;
        const int mirror = radix - p;
        const int butterfly = work % part;
        if constexpr (Planar) {
            if (p == 0) {
                values[butterfly] = src[2 * butterfly];
                values[size + butterfly] = src[2 * butterfly + 1];
                continue;
            }
        }
        const int twist = (butterfly % prev) * outerStep;
        const int a = 2 * (butterfly + p * part);
        const int b = 2 * (butterfly + mirror * part);
        const int wa = 2 * p * twist;
        const int wb = 2 * mirror * twist;
        const float ar = src[a] * roots[wa] - src[a + 1] * roots[wa + 1];
        const float ai = src[a] * roots[wa + 1] + src[a + 1] * roots[wa];
        const float br = src[b] * roots[wb] - src[b + 1] * roots[wb + 1];
        const float bi = src[b] * roots[wb + 1] + src[b + 1] * roots[wb];
        const int sum = elementStride * (butterfly + p * part);
        const int diff = elementStride * (butterfly + mirror * part);
        values[sum] = ar + br;
        values[sum + componentStride] = ai + bi;
        values[diff] = ar - br;
        values[diff + componentStride] = ai - bi;
    }
    asc_syncthreads();

    // In the tiled variant leave DC and any unpaired last frequency here.
    const int singles = TwoFrequencies ? 1 + (half % 2) : half + 1;
    for (int work = threadIdx.x; work < singles * part; work += 256) {
        const int q = TwoFrequencies ? (work / part) * half : work / part;
        const int butterfly = work % part;
        const int output = (butterfly / prev) * (prev * radix) + butterfly % prev;
        float evenRe = values[elementStride * butterfly];
        float evenIm = values[elementStride * butterfly + componentStride];
        if (q == 0) {
            for (int p = 1; p <= half; ++p) {
                const int a = elementStride * (butterfly + p * part);
                evenRe += values[a];
                evenIm += values[a + componentStride];
            }
            result[2 * output] = evenRe;
            result[2 * output + 1] = evenIm;
            continue;
        }
        float oddRe = 0.0f;
        float oddIm = 0.0f;
        const int stride = q * innerStep;
        int w = 0;
        for (int p = 1; p <= half; ++p) {
            w += stride;
            if (w >= rootN)
                w -= rootN;
            // The plan's roots already contain the sign of this transform.
            const float cosine = roots[2 * w];
            const float sine = roots[2 * w + 1];
            const int a = elementStride * (butterfly + p * part);
            const int b = elementStride * (butterfly + (radix - p) * part);
            evenRe += cosine * values[a];
            evenIm += cosine * values[a + componentStride];
            oddRe += sine * values[b];
            oddIm += sine * values[b + componentStride];
        }
        const int first = 2 * (output + q * prev);
        const int second = 2 * (output + (radix - q) * prev);
        result[first] = evenRe - oddIm;
        result[first + 1] = evenIm + oddRe;
        result[second] = evenRe + oddIm;
        result[second + 1] = evenIm - oddRe;
    }
    if constexpr (TwoFrequencies) {
        for (int work = threadIdx.x; work < (half / 2) * part; work += 256) {
            const int q = 1 + 2 * (work / part);
            const int butterfly = work % part;
            const int output = (butterfly / prev) * (prev * radix) + butterfly % prev;
            const float dcRe = values[elementStride * butterfly];
            const float dcIm = values[elementStride * butterfly + componentStride];
            float evenRe[2] = {dcRe, dcRe}, evenIm[2] = {dcIm, dcIm};
            float oddRe[2] = {}, oddIm[2] = {};
            int index[2] = {};
            for (int p = 1; p <= half; ++p) {
                const int a = elementStride * (butterfly + p * part);
                const int b = elementStride * (butterfly + (radix - p) * part);
                const float sumRe = values[a], sumIm = values[a + componentStride];
                const float diffRe = values[b], diffIm = values[b + componentStride];
#pragma unroll
                for (int lane = 0; lane < 2; ++lane) {
                    index[lane] += (q + lane) * innerStep;
                    if (index[lane] >= rootN)
                        index[lane] -= rootN;
                    const float c = roots[2 * index[lane]];
                    const float s = roots[2 * index[lane] + 1];
                    evenRe[lane] += c * sumRe;
                    evenIm[lane] += c * sumIm;
                    oddRe[lane] += s * diffRe;
                    oddIm[lane] += s * diffIm;
                }
            }
#pragma unroll
            for (int lane = 0; lane < 2; ++lane) {
                const int first = 2 * (output + (q + lane) * prev);
                const int second = 2 * (output + (radix - q - lane) * prev);
                result[first] = evenRe[lane] - oddIm[lane];
                result[first + 1] = evenIm[lane] + oddRe[lane];
                result[second] = evenRe[lane] + oddIm[lane];
                result[second + 1] = evenIm[lane] - oddRe[lane];
            }
        }
    }
}
} // namespace fft_mixed
