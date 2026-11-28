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
template <int J, int T, bool SaveOrigin = false>
__simt_vf__ __launch_bounds__(256) inline void Pack(__ubuf__ float* y, __ubuf__ float* roots, __ubuf__ float* packed,
                                                    int group, int block, int prefix, int Q, int rows, int rowStride,
                                                    int rootStep, __ubuf__ float* origin)
{
    constexpr int MH = 2 * J, fragment = MH * T;
    const int H = Q / 2;
    if constexpr (SaveOrigin) {
        // A whole tile has rows * prefix <= J. Only these values are still
        // needed from the prefix after its two packed matrices reach L1.
        const int lane = threadIdx.x;
        if (lane < rows * prefix) {
            const int row = rows == 1 ? 0 : lane / prefix;
            const int pos = row * rowStride + 2 * (lane - row * prefix);
            origin[2 * lane] = y[pos];
            origin[2 * lane + 1] = y[pos + 1];
        }
    }
    for (int i = threadIdx.x; i < J * T; i += 256) {
        int j = i % J, k = i / J, p = block * T + k + 1, lane = group * J + j;
        float sr = 0, si = 0, dr = 0, di = 0;
        if (p <= H && lane < rows * prefix) {
            const int row = rows == 1 ? 0 : lane / prefix, u = lane - row * prefix;
            auto* src = y + row * rowStride;
            int l = 2 * (p * prefix + u), m = 2 * ((Q - p) * prefix + u);
            // Packed even real rows use every second root; full-length FFTs use every root.
            int w = 2 * rootStep * p * u, v = 2 * rootStep * (Q - p) * u;
            float lr = src[l] * roots[w] - src[l + 1] * roots[w + 1];
            float li = src[l] * roots[w + 1] + src[l + 1] * roots[w];
            float mr = src[m] * roots[v] - src[m + 1] * roots[v + 1];
            float mi = src[m] * roots[v + 1] + src[m + 1] * roots[v];
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
template <int J, int T, bool CompactOrigin = false>
__simt_vf__ __launch_bounds__(256) inline void Restore(__ubuf__ float* even, __ubuf__ float* odd, __ubuf__ float* y,
                                                       __ubuf__ float* x, int group, int block, int prefix, int Q,
                                                       int rows, int rowStride)
{
    constexpr int MH = 2 * J;
    const int H = Q / 2;
    for (int i = threadIdx.x; i < J * T; i += 256) {
        int j = i % J, q = block * T + i / J, lane = group * J + j, pos = (i / J) * MH + j;
        if (q > H || lane >= rows * prefix)
            continue;
        const int row = rows == 1 ? 0 : lane / prefix, u = lane - row * prefix;
        auto* src = y + row * (CompactOrigin ? 2 * prefix : rowStride);
        auto* dst = x + row * rowStride;
        float re = src[2 * u] + even[pos], im = src[2 * u + 1] + even[pos + J];
        // The extra cosine column sums the pairs for DC.
        if (q == H) {
            dst[2 * u] = re;
            dst[2 * u + 1] = im;
            continue;
        }
        int k = 2 * ((q + 1) * prefix + u), l = 2 * ((Q - 1 - q) * prefix + u);
        dst[k] = re + odd[pos + J];
        dst[k + 1] = im - odd[pos];
        dst[l] = re - odd[pos + J];
        dst[l + 1] = im + odd[pos];
    }
}
__aicore__ inline int InputOffset(int slot, int k, int stage, int sub, int KB, int fragment)
{
    return ((((slot * KB + k) * 2 + stage) * 2 + sub) * fragment) * sizeof(float);
}
template <int T>
__aicore__ inline void LoadWeightTile(LocalTensor<float> b0, LocalTensor<float> b1, GlobalTensor<float> weights)
{
    DataCopy(b1, weights, Nd2NzParams(T / 16, 16, T, 16 * T, T, 16, 1, 16 * T));
    PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < T / 16; ++i)
        LoadDataWithTranspose(b0[i * 128], b1[i * 256],
                              LoadData2dTransposeParams(0, T / 16, T / 16, T / 8 - 1, T / 16 - 1, inc));
}
template <int J, int T>
__aicore__ inline void Cube(__gm__ float* wg, const __gm__ Tiling* t)
{
    constexpr int MH = 2 * J, M = 2 * MH, fragment = MH * T;
    const int N = t->n, Q = t->tail;
    const bool packedReal = t->pass != 0 && N % 2 == 0;
    const int fftSize = packedReal ? N / 2 : N, prefix = fftSize / Q;
    const int tileRows = t->pass == 0 ? 1 : t->rowsPerAivTile, groups = (tileRows * prefix + J - 1) / J;
    const int KB = t->kBlocks, NB = t->outBlocks, batch = t->rows, rowGroup = t->rowGroup;
    const int rootFloats = (2 * N + 511) / 512 * 512;
    const int rowFloats = packedReal ? (N + 128 + 511) / 512 * 512 : rootFloats;
    const int resultBase = rootFloats + 2 * tileRows * rowFloats + 2 * fragment;
    GlobalTensor<float> weights;
    weights.SetGlobalBuffer(wg);
    LocalTensor<float> b1(TPosition::B1, 128 * 1024, T * T);
    constexpr int weightStride = 4096; // One L0B slot, as in the direct C2C kernel.
    static_assert(T * T <= weightStride);
    LocalTensor<float> b0(TPosition::B2, 0, 2 * weightStride);
    LocalTensor<float> a0(TPosition::A2, 0, M * T);
    LocalTensor<float> c(TPosition::CO1, 0, M * T);
    SetLoadDataPaddingValue<float>(0);
    SetAtomicNone();
    const bool residentWeights = t->pass != 0 && KB == 1 && NB == 1;
    const bool pipeline = residentWeights && groups == 1 && t->realCubePipeline;
    if (residentWeights) {
        for (int stage = 0; stage < 2; ++stage) {
            LoadWeightTile<T>(b0[stage * weightStride], b1, weights[stage * T * T]);
            PipeBarrier<PIPE_ALL>();
        }
    }
    int ordinal = 0;
    for (int pair = 2 * GetBlockIdx() * rowGroup; pair < batch; pair += 2 * GetBlockNum() * rowGroup) {
        for (int inner = 0; inner < rowGroup && pair + inner < batch; inner += tileRows, ++ordinal) {
            for (int g = 0; g < groups; ++g) {
                const int inputSlot = pipeline ? ordinal % 2 : g % 2;
                CrossCoreWaitFlag<2, PIPE_MTE1>(2 + inputSlot);
                for (int n = 0; n < NB; ++n) {
                    const int outputSlot = pipeline ? inputSlot : n % 2;
                    if (n >= 2)
                        CrossCoreWaitFlag<2, PIPE_FIX>(10 + n % 2);
                    for (int stage = 0; stage < 2; ++stage) {
                        for (int k = 0; k < KB; ++k) {
                            if (!residentWeights) {
                                const int offset = ((stage * NB + n) * KB + k) * T * T;
                                LoadWeightTile<T>(b0, b1, weights[offset]);
                            }
                            LocalTensor<float> a1(TPosition::A1, InputOffset(inputSlot, k, stage, 0, KB, fragment),
                                                  M * T);
                            for (int i = 0; i < M / 16; i++)
                                LoadData(a0[i * 128], a1[i * 16 * T],
                                         LoadData2dParams(0, T / 8, 1, 0, M / 16 - 1, false, inc));
                            PipeBarrier<PIPE_ALL>();
                            Mmad(c, a0, b0[residentWeights ? stage * weightStride : 0],
                                 MmadParams(M, T, T, 0, false, k == 0));
                            PipeBarrier<PIPE_ALL>();
                        }
                        LocalTensor<float> recv(TPosition::VECOUT,
                                                (resultBase + (2 * outputSlot + stage) * fragment) * sizeof(float),
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
                    CrossCoreSetFlag<2, PIPE_FIX>(8 + outputSlot);
                }
                if (!pipeline) {
                    for (int n = NB - 2; n < NB; ++n)
                        if (n >= 0)
                            CrossCoreWaitFlag<2, PIPE_FIX>(10 + n % 2);
                }
            }
        }
    }
}
__aicore__ inline void SendPacked(__ubuf__ float* packed, int fragment, int slot, int k, int sub, int kBlocks)
{
    for (int stage = 0; stage < 2; ++stage) {
        LocalTensor<float> dst(TPosition::A1, InputOffset(slot, k, stage, sub, kBlocks, fragment), fragment);
        asc_copy_ub2l1((__cbuf__ float*)dst.GetPhyAddr(), packed + stage * fragment, fragment * sizeof(float));
    }
    PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void WriteTile(__gm__ float* output, __ubuf__ float* data, __ubuf__ float* scratch,
                                 __ubuf__ float* roots, const __gm__ Tiling* t, int row, int tileRows, int rowFloats,
                                 int outFloats)
{
    if (t->pass != 0 && t->n % 2) {
        asc_vf_call<fft_mixed::OddRealBoundary<false>>(dim3(256), data, scratch, t->n, t->pass, tileRows, rowFloats);
        PipeBarrier<PIPE_ALL>();
        if (t->pass == 2)
            data = scratch;
    } else if (t->pass == 1) {
        // Restore every mirrored frequency before the real-spectrum conversion.
        const int n = t->n;
        asc_vf_call<fft_mixed::BatchedRealBoundary>(dim3(256), data, scratch, roots, n, false, tileRows, rowFloats);
        PipeBarrier<PIPE_ALL>();
        data = scratch;
    }
    if (row < t->rows) {
        const int validRows = t->rows - row < tileRows ? t->rows - row : tileRows;
        asc_copy_ub2gm_align(output + outFloats * row, data, validRows, outFloats * 4, 0, outFloats * 4, rowFloats * 4);
    }
    PipeBarrier<PIPE_ALL>();
}
template <int J, int T>
__aicore__ inline void FinishWholeTile(__gm__ float* output, __ubuf__ float* returns, __ubuf__ float* x,
                                       __ubuf__ float* y, __ubuf__ float* roots, const __gm__ Tiling* t, int slot,
                                       int row, int prefix, int tileRows, int rowFloats, int outFloats)
{
    constexpr int fragment = 2 * J * T;
    auto* even = returns + 2 * slot * fragment;
    auto* origin = returns + 4 * fragment + 2 * slot * J;
    const int q = t->tail;
    CrossCoreWaitFlag<2, PIPE_V>(8 + slot);
    asc_vf_call<Restore<J, T, true>>(dim3(256), even, even + fragment, origin, x, 0, 0, prefix, q, tileRows, rowFloats);
    PipeBarrier<PIPE_ALL>();
    WriteTile(output, x, y, roots, t, row, tileRows, rowFloats, outFloats);
}
template <int J, int T>
__aicore__ inline void Vector(__gm__ float* xg, __gm__ float* rg, __gm__ float* yg, const __gm__ Tiling* t)
{
    constexpr int MH = 2 * J, fragment = MH * T;
    const int N = t->n, Q = t->tail, pass = t->pass;
    const bool packedReal = pass != 0 && N % 2 == 0, oddReal = pass != 0 && N % 2;
    const int fftSize = packedReal ? N / 2 : N, prefix = fftSize / Q;
    const int tileRows = pass == 0 ? 1 : t->rowsPerAivTile, groups = (tileRows * prefix + J - 1) / J;
    const int KB = t->kBlocks, NB = t->outBlocks, batch = t->rows, rowGroup = t->rowGroup;
    const int rootFloats = (2 * N + 511) / 512 * 512;
    const int rowFloats = packedReal ? (N + 128 + 511) / 512 * 512 : rootFloats;
    const int resultBase = rootFloats + 2 * tileRows * rowFloats + 2 * fragment;
    const int spectrumFloats = 2 * (N / 2 + 1);
    const int inFloats = pass == 1 ? N : pass == 2 ? spectrumFloats : 2 * N;
    const int outFloats = pass == 1 ? spectrumFloats : pass == 2 ? N : 2 * N;
    const bool pipeline = pass != 0 && KB == 1 && NB == 1 && groups == 1 && t->realCubePipeline;
    __ubuf__ float arena[212 * 1024 / sizeof(float)];
    auto* roots = arena;
    auto* x = roots + rootFloats;
    auto* y = x + tileRows * rowFloats;
    auto* packed = y + tileRows * rowFloats;
    int sub = GetSubBlockIdx();
    asc_copy_gm2ub_align(roots, rg, 1, N * 8, 0, 0, false, 0, 0, 0);
    PipeBarrier<PIPE_ALL>();
    int ordinal = 0, previousRow = 0;
    for (int pair = 2 * (GetBlockIdx() / 2) * rowGroup; pair < batch; pair += 2 * GetBlockNum() * rowGroup) {
        for (int inner = 0; inner < rowGroup && pair + inner < batch; inner += tileRows) {
            // Assign complete 32-byte GM blocks to a single AIV. A missing
            // partner still packs and acknowledges every tile in the final pair.
            int row = pair + sub * rowGroup + inner;
            for (int r = 0; r < tileRows; ++r) {
                const int sourceRow = row + r < batch ? row + r : pair + inner;
                asc_copy_gm2ub_align(x + r * rowFloats, xg + inFloats * sourceRow, 1, inFloats * 4, 0, 0, false, 0, 0,
                                     0);
            }
            PipeBarrier<PIPE_ALL>();
            auto* source = x;
            auto* scratch = y;
            if (oddReal || pass == 2) {
                if (oddReal)
                    asc_vf_call<fft_mixed::OddRealBoundary<true>>(dim3(256), x, y, N, pass, tileRows, rowFloats);
                else
                    asc_vf_call<fft_mixed::BatchedRealBoundary>(dim3(256), x, y, roots, N, true, tileRows, rowFloats);
                PipeBarrier<PIPE_ALL>();
                source = y;
                scratch = x;
            }
            // The real boundary belongs outside the complete prefix + Cube FFT.
            auto* transformed = fft_mixed::ComputeTileButterfly<false, true>(source, scratch, roots, t, tileRows,
                                                                             rowFloats);
            auto* result = transformed == x ? y : x;
            PipeBarrier<PIPE_ALL>();
            if (pipeline) {
                // Both subblocks send and drain in the same tile order, including
                // the dummy partner. Publishing tile i+2 follows draining tile i.
                const int slot = ordinal % 2;
                auto* returns = arena + resultBase;
                auto* origin = returns + 4 * fragment + 2 * slot * J;
                asc_vf_call<Pack<J, T, true>>(dim3(256), transformed, roots, packed, 0, 0, prefix, Q, tileRows,
                                              rowFloats, packedReal ? 2 : 1, origin);
                PipeBarrier<PIPE_ALL>();
                SendPacked(packed, fragment, slot, 0, sub, 1);
                CrossCoreSetFlag<2, PIPE_MTE3>(2 + slot);
                if (ordinal)
                    FinishWholeTile<J, T>(yg, returns, x, y, roots, t, 1 - slot, previousRow, prefix, tileRows,
                                          rowFloats, outFloats);
                previousRow = row;
                ++ordinal;
                continue;
            }
            // Send the next prefix group to L1 before consuming the previous group.
            for (int i = 0; i <= groups; ++i) {
                if (i < groups) {
                    for (int k = 0; k < KB; ++k) {
                        asc_vf_call<Pack<J, T>>(dim3(256), transformed, roots, packed, i, k, prefix, Q, tileRows,
                                                rowFloats, packedReal ? 2 : 1, packed);
                        PipeBarrier<PIPE_ALL>();
                        SendPacked(packed, fragment, i % 2, k, sub, KB);
                    }
                    CrossCoreSetFlag<2, PIPE_MTE3>(2 + i % 2);
                }
                if (i > 0)
                    for (int n = 0; n < NB; ++n) {
                        CrossCoreWaitFlag<2, PIPE_V>(8 + n % 2);
                        auto* e = arena + resultBase + 2 * (n % 2) * fragment;
                        asc_vf_call<Restore<J, T>>(dim3(256), e, e + fragment, transformed, result, i - 1, n, prefix, Q,
                                                   tileRows, rowFloats);
                        PipeBarrier<PIPE_ALL>();
                        CrossCoreSetFlag<2, PIPE_V>(10 + n % 2);
                    }
            }
            WriteTile(yg, result, transformed, roots, t, row, tileRows, rowFloats, outFloats);
        }
    }
    if (ordinal)
        FinishWholeTile<J, T>(yg, arena + resultBase, x, y, roots, t, (ordinal - 1) % 2, previousRow, prefix, tileRows,
                              rowFloats, outFloats);
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
