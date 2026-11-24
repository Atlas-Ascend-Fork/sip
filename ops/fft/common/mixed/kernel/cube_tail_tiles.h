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
#include "row_fft.h"
namespace fft_cube_tiles {
using Tiling = AsdSip::OpParam::MixedCommonTiling;
using namespace AscendC;
// Each AIV owns one half of the rows of both cosine and sine products.
constexpr FixpipeConfig dnUb = {CO2Layout::COLUMN_MAJOR, true};
template <int J, int T>
__simt_vf__ __launch_bounds__(256) inline void Pack(__ubuf__ float* y, __ubuf__ float* roots, __ubuf__ float* packed,
                                                    int group, int block, int prefix, int Q)
{
    constexpr int MH = 2 * J, fragment = MH * T;
    const int H = Q / 2;
    for (int i = threadIdx.x; i < J * T; i += 256) {
        int j = i % J, k = i / J, p = block * T + k + 1, u = group * J + j;
        float sr = 0, si = 0, dr = 0, di = 0;
        if (p <= H && u < prefix) {
            int l = 2 * (p * prefix + u), m = 2 * ((Q - p) * prefix + u), w = 2 * p * u, v = 2 * (Q - p) * u;
            float lr = y[l] * roots[w] - y[l + 1] * roots[w + 1], li = y[l] * roots[w + 1] + y[l + 1] * roots[w];
            float mr = y[m] * roots[v] - y[m + 1] * roots[v + 1], mi = y[m] * roots[v + 1] + y[m + 1] * roots[v];
            sr = lr + mr;
            si = li + mi;
            dr = lr - mr;
            di = li - mi;
        }
        // L0A expects a 16-row by 8-column inner block, with real lanes first.
        int real = (j / 16) * 16 * T + (k / 8) * 128 + (j % 16) * 8 + k % 8;
        int imag = ((J + j) / 16) * 16 * T + (k / 8) * 128 + ((J + j) % 16) * 8 + k % 8;
        packed[real] = sr;
        packed[imag] = si;
        packed[fragment + real] = dr;
        packed[fragment + imag] = di;
    }
}
template <int J, int T>
__simt_vf__ __launch_bounds__(256) inline void Restore(__ubuf__ float* even, __ubuf__ float* odd, __ubuf__ float* y,
                                                       __ubuf__ float* x, int group, int block, int prefix, int Q)
{
    constexpr int MH = 2 * J;
    const int H = Q / 2;
    for (int i = threadIdx.x; i < J * T; i += 256) {
        int j = i % J, q = block * T + i / J, u = group * J + j, pos = (i / J) * MH + j;
        if (q > H || u >= prefix)
            continue;
        float re = y[2 * u] + even[pos], im = y[2 * u + 1] + even[pos + J];
        // The extra cosine column sums the pairs for DC.
        if (q == H) {
            x[2 * u] = re;
            x[2 * u + 1] = im;
            continue;
        }
        int k = 2 * ((q + 1) * prefix + u), l = 2 * ((Q - 1 - q) * prefix + u);
        x[k] = re + odd[pos + J];
        x[k + 1] = im - odd[pos];
        x[l] = re - odd[pos + J];
        x[l + 1] = im + odd[pos];
    }
}
__aicore__ inline int InputOffset(int slot, int k, int stage, int sub, int KB, int fragment)
{
    return ((((slot * KB + k) * 2 + stage) * 2 + sub) * fragment) * sizeof(float);
}
template <int J, int T>
__aicore__ inline void Cube(__gm__ float* wg, const __gm__ Tiling* t)
{
    constexpr int MH = 2 * J, M = 2 * MH, fragment = MH * T;
    const int N = t->n, Q = t->tail, prefix = N / Q, groups = (prefix + J - 1) / J;
    const int KB = t->kBlocks, NB = t->outBlocks, batch = t->rows, rowGroup = t->rowGroup;
    const int rowFloats = (2 * N + 511) / 512 * 512, resultBase = 3 * rowFloats + 2 * fragment;
    GlobalTensor<float> weights;
    weights.SetGlobalBuffer(wg);
    LocalTensor<float> b1(TPosition::B1, 128 * 1024, T * T);
    LocalTensor<float> b0(TPosition::B2, 0, T * T);
    LocalTensor<float> a0(TPosition::A2, 0, M * T);
    LocalTensor<float> c(TPosition::CO1, 0, M * T);
    SetLoadDataPaddingValue<float>(0);
    SetAtomicNone();
    for (int pair = 2 * GetBlockIdx() * rowGroup; pair < batch; pair += 2 * GetBlockNum() * rowGroup) {
        for (int inner = 0; inner < rowGroup && pair + inner < batch; ++inner) {
            for (int g = 0; g < groups; ++g) {
                CrossCoreWaitFlag<2, PIPE_MTE1>(2 + g % 2);
                for (int n = 0; n < NB; ++n) {
                    if (n >= 2)
                        CrossCoreWaitFlag<2, PIPE_FIX>(10 + n % 2);
                    for (int stage = 0; stage < 2; ++stage) {
                        for (int k = 0; k < KB; ++k) {
                            int offset = ((stage * NB + n) * KB + k) * T * T;
                            DataCopy(b1, weights[offset], Nd2NzParams(T / 16, 16, T, 16 * T, T, 16, 1, 16 * T));
                            PipeBarrier<PIPE_ALL>();
                            for (int i = 0; i < T / 16; i++)
                                LoadDataWithTranspose(
                                    b0[i * 128], b1[i * 256],
                                    LoadData2dTransposeParams(0, T / 16, T / 16, T / 8 - 1, T / 16 - 1, inc));
                            LocalTensor<float> a1(TPosition::A1, InputOffset(g % 2, k, stage, 0, KB, fragment), M * T);
                            for (int i = 0; i < M / 16; i++)
                                LoadData(a0[i * 128], a1[i * 16 * T],
                                         LoadData2dParams(0, T / 8, 1, 0, M / 16 - 1, false, inc));
                            PipeBarrier<PIPE_ALL>();
                            Mmad(c, a0, b0, MmadParams(M, T, T, 0, false, k == 0));
                            PipeBarrier<PIPE_ALL>();
                        }
                        LocalTensor<float> recv(TPosition::VECOUT,
                                                (resultBase + (2 * (n % 2) + stage) * fragment) * sizeof(float),
                                                fragment);
                        FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fp;
                        fp.nSize = T;
                        fp.mSize = MH;
                        fp.srcStride = M;
                        fp.dstStride = MH;
                        fp.quantPre = QuantMode_t::NoQuant;
                        fp.params.dnNum = 1;
                        fp.params.srcNzC0Stride = 1;
                        fp.params.srcNzMatrixStride = 0;
                        fp.params.dstDnMatrixStride = 0;
                        fp.subBlockId = 0;
                        Fixpipe<float, float, dnUb>(recv, c, fp);
                        fp.subBlockId = 1;
                        Fixpipe<float, float, dnUb>(recv, c[MH * 16], fp);
                        PipeBarrier<PIPE_ALL>();
                    }
                    CrossCoreSetFlag<2, PIPE_FIX>(8 + n % 2);
                }
                for (int n = NB - 2; n < NB; ++n)
                    if (n >= 0)
                        CrossCoreWaitFlag<2, PIPE_FIX>(10 + n % 2);
            }
        }
    }
}
template <int J, int T>
__aicore__ inline void Vector(__gm__ float* xg, __gm__ float* rg, __gm__ float* yg, const __gm__ Tiling* t)
{
    constexpr int MH = 2 * J, fragment = MH * T;
    const int N = t->n, Q = t->tail, prefix = N / Q, groups = (prefix + J - 1) / J;
    const int KB = t->kBlocks, NB = t->outBlocks, batch = t->rows, rowGroup = t->rowGroup;
    const int rowFloats = (2 * N + 511) / 512 * 512, resultBase = 3 * rowFloats + 2 * fragment;
    __ubuf__ float arena[212 * 1024 / sizeof(float)];
    auto* roots = arena;
    auto* x = roots + rowFloats;
    auto* y = x + rowFloats;
    auto* packed = y + rowFloats;
    int sub = GetSubBlockIdx();
    asc_copy_gm2ub_align(roots, rg, 1, N * 8, 0, 0, false, 0, 0, 0);
    PipeBarrier<PIPE_ALL>();
    for (int pair = 2 * (GetBlockIdx() / 2) * rowGroup; pair < batch; pair += 2 * GetBlockNum() * rowGroup) {
        for (int inner = 0; inner < rowGroup && pair + inner < batch; ++inner) {
            // Assign complete 32-byte GM blocks to a single AIV. A missing
            // partner still packs and acknowledges every tile in the final pair.
            int row = pair + sub * rowGroup + inner;
            asc_copy_gm2ub_align(x, xg + 2 * N * (row < batch ? row : pair + inner), 1, N * 8, 0, 0, false, 0, 0, 0);
            PipeBarrier<PIPE_ALL>();
            auto* transformed = fft_mixed::ComputeTileButterfly(x, y, roots, t, 1, rowFloats);
            auto* result = transformed == x ? y : x;
            PipeBarrier<PIPE_ALL>();
            // Send the next prefix group to L1 before consuming the previous group.
            for (int i = 0; i <= groups; ++i) {
                if (i < groups) {
                    for (int k = 0; k < KB; ++k) {
                        asc_vf_call<Pack<J, T>>(dim3(256), transformed, roots, packed, i, k, prefix, Q);
                        PipeBarrier<PIPE_ALL>();
                        for (int s = 0; s < 2; ++s) {
                            LocalTensor<float> dst(TPosition::A1, InputOffset(i % 2, k, s, sub, KB, fragment),
                                                   fragment);
                            asc_copy_ub2l1((__cbuf__ float*)dst.GetPhyAddr(), packed + s * fragment,
                                           fragment * sizeof(float));
                        }
                        PipeBarrier<PIPE_ALL>();
                    }
                    CrossCoreSetFlag<2, PIPE_MTE3>(2 + i % 2);
                }
                if (i > 0)
                    for (int n = 0; n < NB; ++n) {
                        CrossCoreWaitFlag<2, PIPE_V>(8 + n % 2);
                        auto* e = arena + resultBase + 2 * (n % 2) * fragment;
                        asc_vf_call<Restore<J, T>>(dim3(256), e, e + fragment, transformed, result, i - 1, n, prefix,
                                                   Q);
                        PipeBarrier<PIPE_ALL>();
                        CrossCoreSetFlag<2, PIPE_V>(10 + n % 2);
                    }
            }
            if (row < batch)
                asc_copy_ub2gm_align(yg + 2 * N * row, result, 1, N * 8, 0, 0, 0);
            PipeBarrier<PIPE_ALL>();
        }
    }
}
template <int J, int T>
__aicore__ inline void Run(__gm__ float* x, __gm__ float* roots, __gm__ float* y, const __gm__ Tiling* t)
{
    if ASCEND_IS_AIC {
        Cube<J, T>(roots + (2 * t->n + 7) / 8 * 8, t);
    }
    if ASCEND_IS_AIV {
        Vector<J, T>(x, roots, y, t);
    }
}
} // namespace fft_cube_tiles
