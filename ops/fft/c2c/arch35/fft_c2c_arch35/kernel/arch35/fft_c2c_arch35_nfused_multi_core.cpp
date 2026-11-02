/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// N fused transform kernel (design point 2): 500=20x25 two-level register fusion, butterflies ported from example
// fft_simt_ops.h
//   Example-isomorphic: input DataCopy into UB tile, inter-stage data stays in UB+registers (asc_syncthreads), scatter
//   back to GM. See handover doc ch14
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace {
struct NFusedTilingData {
    int64_t batch;
    int32_t fftM;
    int32_t direction; // 1=forward
    int32_t twFloatOff;
    int32_t coreNum;
};

constexpr int32_t NF_SIMT_THREADS = 256;
constexpr int32_t NF_ROW_FLOATS = 1000; // 500 婢跺秵鏆?= 1000 濞搭喚鍋?

constexpr float R5_C72 = 0.30901699437494745f;
constexpr float R5_C144 = -0.80901699494794734f;
constexpr float R5_S72 = 0.95105651629515357f;
constexpr float R5_S144 = 0.58778525229247314f;

constexpr int32_t R4_T1_OFF = 0;
constexpr int32_t R4_T2_OFF = 250;
constexpr int32_t R4_T3_OFF = 500;
constexpr int32_t TW3_OFF = 750;
constexpr int32_t TW4_OFF = 950;

#define NFD5(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, sgn)               \
    do {                                                                          \
        const float s1r = (x1r) + (x4r), s1i = (x1i) + (x4i);                     \
        const float s2r = (x2r) + (x3r), s2i = (x2i) + (x3i);                     \
        const float d1r = (x1r) - (x4r), d1i = (x1i) - (x4i);                     \
        const float d2r = (x2r) - (x3r), d2i = (x2i) - (x3i);                     \
        const float A1 = R5_C72 * s1r + R5_C144 * s2r;                            \
        const float B1 = R5_C72 * s1i + R5_C144 * s2i;                            \
        const float D1 = R5_S72 * d1r + R5_S144 * d2r;                            \
        const float E1 = R5_S72 * d1i + R5_S144 * d2i;                            \
        const float A2 = R5_C144 * s1r + R5_C72 * s2r;                            \
        const float B2 = R5_C144 * s1i + R5_C72 * s2i;                            \
        const float D2 = R5_S144 * d1r - R5_S72 * d2r;                            \
        const float E2 = R5_S144 * d1i - R5_S72 * d2i;                            \
        const float t0r = (x0r) + s1r + s2r, t0i = (x0i) + s1i + s2i;             \
        const float t1r = (x0r) + A1 + (sgn) * E1, t1i = (x0i) + B1 - (sgn) * D1; \
        const float t4r = (x0r) + A1 - (sgn) * E1, t4i = (x0i) + B1 + (sgn) * D1; \
        const float t2r = (x0r) + A2 + (sgn) * E2, t2i = (x0i) + B2 - (sgn) * D2; \
        const float t3r = (x0r) + A2 - (sgn) * E2, t3i = (x0i) + B2 + (sgn) * D2; \
        (x0r) = t0r;                                                              \
        (x0i) = t0i;                                                              \
        (x1r) = t1r;                                                              \
        (x1i) = t1i;                                                              \
        (x4r) = t4r;                                                              \
        (x4i) = t4i;                                                              \
        (x2r) = t2r;                                                              \
        (x2i) = t2i;                                                              \
        (x3r) = t3r;                                                              \
        (x3i) = t3i;                                                              \
    } while (0)

#define NFR4(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, w1r, w1i, w2r, w2i, w3r, w3i, sgn) \
    do {                                                                                \
        const float a0r = (x0r) + (x2r), a0i = (x0i) + (x2i);                           \
        const float a1r = (x0r) - (x2r), a1i = (x0i) - (x2i);                           \
        const float a2r = (x1r) + (x3r), a2i = (x1i) + (x3i);                           \
        const float a3r = (x1r) - (x3r), a3i = (x1i) - (x3i);                           \
        const float y0r = a0r + a2r, y0i = a0i + a2i;                                   \
        const float b1r = a1r + (sgn) * a3i, b1i = a1i - (sgn) * a3r;                   \
        const float b3r = a1r - (sgn) * a3i, b3i = a1i + (sgn) * a3r;                   \
        const float b2r = a0r - a2r, b2i = a0i - a2i;                                   \
        const float y1r = b1r * (w1r) - b1i * (w1i), y1i = b1r * (w1i) + b1i * (w1r);   \
        const float y2r = b2r * (w2r) - b2i * (w2i), y2i = b2r * (w2i) + b2i * (w2r);   \
        const float y3r = b3r * (w3r) - b3i * (w3i), y3i = b3r * (w3i) + b3i * (w3r);   \
        (x0r) = y0r;                                                                    \
        (x0i) = y0i;                                                                    \
        (x1r) = y1r;                                                                    \
        (x1i) = y1i;                                                                    \
        (x2r) = y2r;                                                                    \
        (x2i) = y2i;                                                                    \
        (x3r) = y3r;                                                                    \
        (x3i) = y3i;                                                                    \
    } while (0)

#define NFTWMUL(cr, ci, tr, ti)                     \
    do {                                            \
        const float zr = (cr) * (tr) - (ci) * (ti); \
        const float zi = (cr) * (ti) + (ci) * (tr); \
        (cr) = zr;                                  \
        (ci) = zi;                                  \
    } while (0)

#define NFDIGREV20(f) ((f) / 5 + 4 * ((f) % 5))
} // namespace

// 閿?A:radix-20(r4 D=125 閿?r5 D=25);閿?B:radix-25 閾诲秵鏆熺€涙寮芥潪顒佹殠閿?
#define NF_STAGE20_UB(base0, fftM, sgn)                                                                 \
    do {                                                                                                \
        const uint32_t total = 25 * static_cast<uint32_t>(fftM);                                        \
        const uint32_t step25 = 50;                                                                     \
        const uint32_t iters = (total + NF_SIMT_THREADS - 1) / NF_SIMT_THREADS;                         \
        for (uint32_t it = 0; it < iters; ++it) {                                                       \
            const uint32_t e = it * NF_SIMT_THREADS + threadIdx.x;                                      \
            if (e >= total)                                                                             \
                continue;                                                                               \
            const uint32_t r = e % static_cast<uint32_t>(fftM);                                         \
            const uint32_t g = e / static_cast<uint32_t>(fftM);                                         \
            const uint32_t base = (base0) + r * NF_ROW_FLOATS + 2 * g;                                  \
            float x0r = tile[base];                                                                     \
            float x0i = tile[base + 1];                                                                 \
            float x1r = tile[base + step25];                                                            \
            float x1i = tile[base + step25 + 1];                                                        \
            float x2r = tile[base + 2 * step25];                                                        \
            float x2i = tile[base + 2 * step25 + 1];                                                    \
            float x3r = tile[base + 3 * step25];                                                        \
            float x3i = tile[base + 3 * step25 + 1];                                                    \
            float x4r = tile[base + 4 * step25];                                                        \
            float x4i = tile[base + 4 * step25 + 1];                                                    \
            float x5r = tile[base + 5 * step25];                                                        \
            float x5i = tile[base + 5 * step25 + 1];                                                    \
            float x6r = tile[base + 6 * step25];                                                        \
            float x6i = tile[base + 6 * step25 + 1];                                                    \
            float x7r = tile[base + 7 * step25];                                                        \
            float x7i = tile[base + 7 * step25 + 1];                                                    \
            float x8r = tile[base + 8 * step25];                                                        \
            float x8i = tile[base + 8 * step25 + 1];                                                    \
            float x9r = tile[base + 9 * step25];                                                        \
            float x9i = tile[base + 9 * step25 + 1];                                                    \
            float x10r = tile[base + 10 * step25];                                                      \
            float x10i = tile[base + 10 * step25 + 1];                                                  \
            float x11r = tile[base + 11 * step25];                                                      \
            float x11i = tile[base + 11 * step25 + 1];                                                  \
            float x12r = tile[base + 12 * step25];                                                      \
            float x12i = tile[base + 12 * step25 + 1];                                                  \
            float x13r = tile[base + 13 * step25];                                                      \
            float x13i = tile[base + 13 * step25 + 1];                                                  \
            float x14r = tile[base + 14 * step25];                                                      \
            float x14i = tile[base + 14 * step25 + 1];                                                  \
            float x15r = tile[base + 15 * step25];                                                      \
            float x15i = tile[base + 15 * step25 + 1];                                                  \
            float x16r = tile[base + 16 * step25];                                                      \
            float x16i = tile[base + 16 * step25 + 1];                                                  \
            float x17r = tile[base + 17 * step25];                                                      \
            float x17i = tile[base + 17 * step25 + 1];                                                  \
            float x18r = tile[base + 18 * step25];                                                      \
            float x18i = tile[base + 18 * step25 + 1];                                                  \
            float x19r = tile[base + 19 * step25];                                                      \
            float x19i = tile[base + 19 * step25 + 1];                                                  \
            {                                                                                           \
                const uint32_t j0 = g + 25 * 0;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j0], w1i = tw[(R4_T1_OFF) + 2 * j0 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j0], w2i = tw[(R4_T2_OFF) + 2 * j0 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j0], w3i = tw[(R4_T3_OFF) + 2 * j0 + 1];               \
                NFR4(x0r, x0i, x5r, x5i, x10r, x10i, x15r, x15i, w1r, w1i, w2r, w2i, w3r, w3i, sgn);    \
            }                                                                                           \
            {                                                                                           \
                const uint32_t j1 = g + 25 * 1;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j1], w1i = tw[(R4_T1_OFF) + 2 * j1 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j1], w2i = tw[(R4_T2_OFF) + 2 * j1 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j1], w3i = tw[(R4_T3_OFF) + 2 * j1 + 1];               \
                NFR4(x1r, x1i, x6r, x6i, x11r, x11i, x16r, x16i, w1r, w1i, w2r, w2i, w3r, w3i, sgn);    \
            }                                                                                           \
            {                                                                                           \
                const uint32_t j2 = g + 25 * 2;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j2], w1i = tw[(R4_T1_OFF) + 2 * j2 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j2], w2i = tw[(R4_T2_OFF) + 2 * j2 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j2], w3i = tw[(R4_T3_OFF) + 2 * j2 + 1];               \
                NFR4(x2r, x2i, x7r, x7i, x12r, x12i, x17r, x17i, w1r, w1i, w2r, w2i, w3r, w3i, sgn);    \
            }                                                                                           \
            {                                                                                           \
                const uint32_t j3 = g + 25 * 3;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j3], w1i = tw[(R4_T1_OFF) + 2 * j3 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j3], w2i = tw[(R4_T2_OFF) + 2 * j3 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j3], w3i = tw[(R4_T3_OFF) + 2 * j3 + 1];               \
                NFR4(x3r, x3i, x8r, x8i, x13r, x13i, x18r, x18i, w1r, w1i, w2r, w2i, w3r, w3i, sgn);    \
            }                                                                                           \
            {                                                                                           \
                const uint32_t j4 = g + 25 * 4;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j4], w1i = tw[(R4_T1_OFF) + 2 * j4 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j4], w2i = tw[(R4_T2_OFF) + 2 * j4 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j4], w3i = tw[(R4_T3_OFF) + 2 * j4 + 1];               \
                NFR4(x4r, x4i, x9r, x9i, x14r, x14i, x19r, x19i, w1r, w1i, w2r, w2i, w3r, w3i, sgn);    \
            }                                                                                           \
            {                                                                                           \
                float k1r = tw[(TW3_OFF) + 2 * (g * 4 + 0)], k1i = tw[(TW3_OFF) + 2 * (g * 4 + 0) + 1]; \
                float k2r = tw[(TW3_OFF) + 2 * (g * 4 + 1)], k2i = tw[(TW3_OFF) + 2 * (g * 4 + 1) + 1]; \
                float k3r = tw[(TW3_OFF) + 2 * (g * 4 + 2)], k3i = tw[(TW3_OFF) + 2 * (g * 4 + 2) + 1]; \
                float k4r = tw[(TW3_OFF) + 2 * (g * 4 + 3)], k4i = tw[(TW3_OFF) + 2 * (g * 4 + 3) + 1]; \
                NFD5(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, sgn);                            \
                NFTWMUL(x1r, x1i, k1r, k1i);                                                            \
                NFTWMUL(x4r, x4i, k4r, k4i);                                                            \
                NFTWMUL(x2r, x2i, k2r, k2i);                                                            \
                NFTWMUL(x3r, x3i, k3r, k3i);                                                            \
            }                                                                                           \
            {                                                                                           \
                float k1r = tw[(TW3_OFF) + 2 * (g * 4 + 0)], k1i = tw[(TW3_OFF) + 2 * (g * 4 + 0) + 1]; \
                float k2r = tw[(TW3_OFF) + 2 * (g * 4 + 1)], k2i = tw[(TW3_OFF) + 2 * (g * 4 + 1) + 1]; \
                float k3r = tw[(TW3_OFF) + 2 * (g * 4 + 2)], k3i = tw[(TW3_OFF) + 2 * (g * 4 + 2) + 1]; \
                float k4r = tw[(TW3_OFF) + 2 * (g * 4 + 3)], k4i = tw[(TW3_OFF) + 2 * (g * 4 + 3) + 1]; \
                NFD5(x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i, sgn);                            \
                NFTWMUL(x6r, x6i, k1r, k1i);                                                            \
                NFTWMUL(x9r, x9i, k4r, k4i);                                                            \
                NFTWMUL(x7r, x7i, k2r, k2i);                                                            \
                NFTWMUL(x8r, x8i, k3r, k3i);                                                            \
            }                                                                                           \
            {                                                                                           \
                float k1r = tw[(TW3_OFF) + 2 * (g * 4 + 0)], k1i = tw[(TW3_OFF) + 2 * (g * 4 + 0) + 1]; \
                float k2r = tw[(TW3_OFF) + 2 * (g * 4 + 1)], k2i = tw[(TW3_OFF) + 2 * (g * 4 + 1) + 1]; \
                float k3r = tw[(TW3_OFF) + 2 * (g * 4 + 2)], k3i = tw[(TW3_OFF) + 2 * (g * 4 + 2) + 1]; \
                float k4r = tw[(TW3_OFF) + 2 * (g * 4 + 3)], k4i = tw[(TW3_OFF) + 2 * (g * 4 + 3) + 1]; \
                NFD5(x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i, sgn);                  \
                NFTWMUL(x11r, x11i, k1r, k1i);                                                          \
                NFTWMUL(x14r, x14i, k4r, k4i);                                                          \
                NFTWMUL(x12r, x12i, k2r, k2i);                                                          \
                NFTWMUL(x13r, x13i, k3r, k3i);                                                          \
            }                                                                                           \
            {                                                                                           \
                float k1r = tw[(TW3_OFF) + 2 * (g * 4 + 0)], k1i = tw[(TW3_OFF) + 2 * (g * 4 + 0) + 1]; \
                float k2r = tw[(TW3_OFF) + 2 * (g * 4 + 1)], k2i = tw[(TW3_OFF) + 2 * (g * 4 + 1) + 1]; \
                float k3r = tw[(TW3_OFF) + 2 * (g * 4 + 2)], k3i = tw[(TW3_OFF) + 2 * (g * 4 + 2) + 1]; \
                float k4r = tw[(TW3_OFF) + 2 * (g * 4 + 3)], k4i = tw[(TW3_OFF) + 2 * (g * 4 + 3) + 1]; \
                NFD5(x15r, x15i, x16r, x16i, x17r, x17i, x18r, x18i, x19r, x19i, sgn);                  \
                NFTWMUL(x16r, x16i, k1r, k1i);                                                          \
                NFTWMUL(x19r, x19i, k4r, k4i);                                                          \
                NFTWMUL(x17r, x17i, k2r, k2i);                                                          \
                NFTWMUL(x18r, x18i, k3r, k3i);                                                          \
            }                                                                                           \
            tile[base] = x0r;                                                                           \
            tile[base + 1] = x0i;                                                                       \
            tile[base + step25] = x1r;                                                                  \
            tile[base + step25 + 1] = x1i;                                                              \
            tile[base + 2 * step25] = x2r;                                                              \
            tile[base + 2 * step25 + 1] = x2i;                                                          \
            tile[base + 3 * step25] = x3r;                                                              \
            tile[base + 3 * step25 + 1] = x3i;                                                          \
            tile[base + 4 * step25] = x4r;                                                              \
            tile[base + 4 * step25 + 1] = x4i;                                                          \
            tile[base + 5 * step25] = x5r;                                                              \
            tile[base + 5 * step25 + 1] = x5i;                                                          \
            tile[base + 6 * step25] = x6r;                                                              \
            tile[base + 6 * step25 + 1] = x6i;                                                          \
            tile[base + 7 * step25] = x7r;                                                              \
            tile[base + 7 * step25 + 1] = x7i;                                                          \
            tile[base + 8 * step25] = x8r;                                                              \
            tile[base + 8 * step25 + 1] = x8i;                                                          \
            tile[base + 9 * step25] = x9r;                                                              \
            tile[base + 9 * step25 + 1] = x9i;                                                          \
            tile[base + 10 * step25] = x10r;                                                            \
            tile[base + 10 * step25 + 1] = x10i;                                                        \
            tile[base + 11 * step25] = x11r;                                                            \
            tile[base + 11 * step25 + 1] = x11i;                                                        \
            tile[base + 12 * step25] = x12r;                                                            \
            tile[base + 12 * step25 + 1] = x12i;                                                        \
            tile[base + 13 * step25] = x13r;                                                            \
            tile[base + 13 * step25 + 1] = x13i;                                                        \
            tile[base + 14 * step25] = x14r;                                                            \
            tile[base + 14 * step25 + 1] = x14i;                                                        \
            tile[base + 15 * step25] = x15r;                                                            \
            tile[base + 15 * step25 + 1] = x15i;                                                        \
            tile[base + 16 * step25] = x16r;                                                            \
            tile[base + 16 * step25 + 1] = x16i;                                                        \
            tile[base + 17 * step25] = x17r;                                                            \
            tile[base + 17 * step25 + 1] = x17i;                                                        \
            tile[base + 18 * step25] = x18r;                                                            \
            tile[base + 18 * step25 + 1] = x18i;                                                        \
            tile[base + 19 * step25] = x19r;                                                            \
            tile[base + 19 * step25 + 1] = x19i;                                                        \
        }                                                                                               \
    } while (0)

#define NF_STAGE25_SCATTER(baseIn, baseOut, fftM, sgn)                                                  \
    do {                                                                                                \
        const uint32_t total = 20 * static_cast<uint32_t>(fftM);                                        \
        const uint32_t step = 2;                                                                        \
        const uint32_t dk3 = 20 * static_cast<uint32_t>(fftM);                                          \
        const uint32_t dk4 = 100 * static_cast<uint32_t>(fftM);                                         \
        const uint32_t iters = (total + NF_SIMT_THREADS - 1) / NF_SIMT_THREADS;                         \
        for (uint32_t it = 0; it < iters; ++it) {                                                       \
            const uint32_t e = it * NF_SIMT_THREADS + threadIdx.x;                                      \
            if (e >= total)                                                                             \
                continue;                                                                               \
            const uint32_t r = e % static_cast<uint32_t>(fftM);                                         \
            const uint32_t fp = e / static_cast<uint32_t>(fftM);                                        \
            const uint32_t base = (baseIn) + r * NF_ROW_FLOATS + 50 * fp;                               \
            float x0r = tile[base];                                                                     \
            float x0i = tile[base + 1];                                                                 \
            float x1r = tile[base + step];                                                              \
            float x1i = tile[base + step + 1];                                                          \
            float x2r = tile[base + 2 * step];                                                          \
            float x2i = tile[base + 2 * step + 1];                                                      \
            float x3r = tile[base + 3 * step];                                                          \
            float x3i = tile[base + 3 * step + 1];                                                      \
            float x4r = tile[base + 4 * step];                                                          \
            float x4i = tile[base + 4 * step + 1];                                                      \
            float x5r = tile[base + 5 * step];                                                          \
            float x5i = tile[base + 5 * step + 1];                                                      \
            float x6r = tile[base + 6 * step];                                                          \
            float x6i = tile[base + 6 * step + 1];                                                      \
            float x7r = tile[base + 7 * step];                                                          \
            float x7i = tile[base + 7 * step + 1];                                                      \
            float x8r = tile[base + 8 * step];                                                          \
            float x8i = tile[base + 8 * step + 1];                                                      \
            float x9r = tile[base + 9 * step];                                                          \
            float x9i = tile[base + 9 * step + 1];                                                      \
            float x10r = tile[base + 10 * step];                                                        \
            float x10i = tile[base + 10 * step + 1];                                                    \
            float x11r = tile[base + 11 * step];                                                        \
            float x11i = tile[base + 11 * step + 1];                                                    \
            float x12r = tile[base + 12 * step];                                                        \
            float x12i = tile[base + 12 * step + 1];                                                    \
            float x13r = tile[base + 13 * step];                                                        \
            float x13i = tile[base + 13 * step + 1];                                                    \
            float x14r = tile[base + 14 * step];                                                        \
            float x14i = tile[base + 14 * step + 1];                                                    \
            float x15r = tile[base + 15 * step];                                                        \
            float x15i = tile[base + 15 * step + 1];                                                    \
            float x16r = tile[base + 16 * step];                                                        \
            float x16i = tile[base + 16 * step + 1];                                                    \
            float x17r = tile[base + 17 * step];                                                        \
            float x17i = tile[base + 17 * step + 1];                                                    \
            float x18r = tile[base + 18 * step];                                                        \
            float x18i = tile[base + 18 * step + 1];                                                    \
            float x19r = tile[base + 19 * step];                                                        \
            float x19i = tile[base + 19 * step + 1];                                                    \
            float x20r = tile[base + 20 * step];                                                        \
            float x20i = tile[base + 20 * step + 1];                                                    \
            float x21r = tile[base + 21 * step];                                                        \
            float x21i = tile[base + 21 * step + 1];                                                    \
            float x22r = tile[base + 22 * step];                                                        \
            float x22i = tile[base + 22 * step + 1];                                                    \
            float x23r = tile[base + 23 * step];                                                        \
            float x23i = tile[base + 23 * step + 1];                                                    \
            float x24r = tile[base + 24 * step];                                                        \
            float x24i = tile[base + 24 * step + 1];                                                    \
            {                                                                                           \
                float k1r = tw[(TW4_OFF) + 2 * (0 * 4 + 0)], k1i = tw[(TW4_OFF) + 2 * (0 * 4 + 0) + 1]; \
                float k2r = tw[(TW4_OFF) + 2 * (0 * 4 + 1)], k2i = tw[(TW4_OFF) + 2 * (0 * 4 + 1) + 1]; \
                float k3r = tw[(TW4_OFF) + 2 * (0 * 4 + 2)], k3i = tw[(TW4_OFF) + 2 * (0 * 4 + 2) + 1]; \
                float k4r = tw[(TW4_OFF) + 2 * (0 * 4 + 3)], k4i = tw[(TW4_OFF) + 2 * (0 * 4 + 3) + 1]; \
                NFD5(x0r, x0i, x5r, x5i, x10r, x10i, x15r, x15i, x20r, x20i, sgn);                      \
                NFTWMUL(x5r, x5i, k1r, k1i);                                                            \
                NFTWMUL(x20r, x20i, k4r, k4i);                                                          \
                NFTWMUL(x10r, x10i, k2r, k2i);                                                          \
                NFTWMUL(x15r, x15i, k3r, k3i);                                                          \
            }                                                                                           \
            {                                                                                           \
                float k1r = tw[(TW4_OFF) + 2 * (1 * 4 + 0)], k1i = tw[(TW4_OFF) + 2 * (1 * 4 + 0) + 1]; \
                float k2r = tw[(TW4_OFF) + 2 * (1 * 4 + 1)], k2i = tw[(TW4_OFF) + 2 * (1 * 4 + 1) + 1]; \
                float k3r = tw[(TW4_OFF) + 2 * (1 * 4 + 2)], k3i = tw[(TW4_OFF) + 2 * (1 * 4 + 2) + 1]; \
                float k4r = tw[(TW4_OFF) + 2 * (1 * 4 + 3)], k4i = tw[(TW4_OFF) + 2 * (1 * 4 + 3) + 1]; \
                NFD5(x1r, x1i, x6r, x6i, x11r, x11i, x16r, x16i, x21r, x21i, sgn);                      \
                NFTWMUL(x6r, x6i, k1r, k1i);                                                            \
                NFTWMUL(x21r, x21i, k4r, k4i);                                                          \
                NFTWMUL(x11r, x11i, k2r, k2i);                                                          \
                NFTWMUL(x16r, x16i, k3r, k3i);                                                          \
            }                                                                                           \
            {                                                                                           \
                float k1r = tw[(TW4_OFF) + 2 * (2 * 4 + 0)], k1i = tw[(TW4_OFF) + 2 * (2 * 4 + 0) + 1]; \
                float k2r = tw[(TW4_OFF) + 2 * (2 * 4 + 1)], k2i = tw[(TW4_OFF) + 2 * (2 * 4 + 1) + 1]; \
                float k3r = tw[(TW4_OFF) + 2 * (2 * 4 + 2)], k3i = tw[(TW4_OFF) + 2 * (2 * 4 + 2) + 1]; \
                float k4r = tw[(TW4_OFF) + 2 * (2 * 4 + 3)], k4i = tw[(TW4_OFF) + 2 * (2 * 4 + 3) + 1]; \
                NFD5(x2r, x2i, x7r, x7i, x12r, x12i, x17r, x17i, x22r, x22i, sgn);                      \
                NFTWMUL(x7r, x7i, k1r, k1i);                                                            \
                NFTWMUL(x22r, x22i, k4r, k4i);                                                          \
                NFTWMUL(x12r, x12i, k2r, k2i);                                                          \
                NFTWMUL(x17r, x17i, k3r, k3i);                                                          \
            }                                                                                           \
            {                                                                                           \
                float k1r = tw[(TW4_OFF) + 2 * (3 * 4 + 0)], k1i = tw[(TW4_OFF) + 2 * (3 * 4 + 0) + 1]; \
                float k2r = tw[(TW4_OFF) + 2 * (3 * 4 + 1)], k2i = tw[(TW4_OFF) + 2 * (3 * 4 + 1) + 1]; \
                float k3r = tw[(TW4_OFF) + 2 * (3 * 4 + 2)], k3i = tw[(TW4_OFF) + 2 * (3 * 4 + 2) + 1]; \
                float k4r = tw[(TW4_OFF) + 2 * (3 * 4 + 3)], k4i = tw[(TW4_OFF) + 2 * (3 * 4 + 3) + 1]; \
                NFD5(x3r, x3i, x8r, x8i, x13r, x13i, x18r, x18i, x23r, x23i, sgn);                      \
                NFTWMUL(x8r, x8i, k1r, k1i);                                                            \
                NFTWMUL(x23r, x23i, k4r, k4i);                                                          \
                NFTWMUL(x13r, x13i, k2r, k2i);                                                          \
                NFTWMUL(x18r, x18i, k3r, k3i);                                                          \
            }                                                                                           \
            {                                                                                           \
                float k1r = tw[(TW4_OFF) + 2 * (4 * 4 + 0)], k1i = tw[(TW4_OFF) + 2 * (4 * 4 + 0) + 1]; \
                float k2r = tw[(TW4_OFF) + 2 * (4 * 4 + 1)], k2i = tw[(TW4_OFF) + 2 * (4 * 4 + 1) + 1]; \
                float k3r = tw[(TW4_OFF) + 2 * (4 * 4 + 2)], k3i = tw[(TW4_OFF) + 2 * (4 * 4 + 2) + 1]; \
                float k4r = tw[(TW4_OFF) + 2 * (4 * 4 + 3)], k4i = tw[(TW4_OFF) + 2 * (4 * 4 + 3) + 1]; \
                NFD5(x4r, x4i, x9r, x9i, x14r, x14i, x19r, x19i, x24r, x24i, sgn);                      \
                NFTWMUL(x9r, x9i, k1r, k1i);                                                            \
                NFTWMUL(x24r, x24i, k4r, k4i);                                                          \
                NFTWMUL(x14r, x14i, k2r, k2i);                                                          \
                NFTWMUL(x19r, x19i, k3r, k3i);                                                          \
            }                                                                                           \
            NFD5(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, sgn);                                \
            NFD5(x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i, sgn);                                \
            NFD5(x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i, sgn);                      \
            NFD5(x15r, x15i, x16r, x16i, x17r, x17i, x18r, x18i, x19r, x19i, sgn);                      \
            NFD5(x20r, x20i, x21r, x21i, x22r, x22i, x23r, x23i, x24r, x24i, sgn);                      \
            const uint32_t fr = NFDIGREV20(fp);                                                         \
            const uint32_t gmBase = (baseOut) + fr * static_cast<uint32_t>(fftM) + r;                   \
            __gm__ float2* d2 = (__gm__ float2*)(out);                                                  \
            d2[gmBase] = make_float2(x0r, x0i);                                                         \
            d2[gmBase + dk4] = make_float2(x1r, x1i);                                                   \
            d2[gmBase + 2 * dk4] = make_float2(x2r, x2i);                                               \
            d2[gmBase + 3 * dk4] = make_float2(x3r, x3i);                                               \
            d2[gmBase + 4 * dk4] = make_float2(x4r, x4i);                                               \
            d2[gmBase + dk3] = make_float2(x5r, x5i);                                                   \
            d2[gmBase + dk3 + dk4] = make_float2(x6r, x6i);                                             \
            d2[gmBase + dk3 + 2 * dk4] = make_float2(x7r, x7i);                                         \
            d2[gmBase + dk3 + 3 * dk4] = make_float2(x8r, x8i);                                         \
            d2[gmBase + dk3 + 4 * dk4] = make_float2(x9r, x9i);                                         \
            d2[gmBase + 2 * dk3] = make_float2(x10r, x10i);                                             \
            d2[gmBase + 2 * dk3 + dk4] = make_float2(x11r, x11i);                                       \
            d2[gmBase + 2 * dk3 + 2 * dk4] = make_float2(x12r, x12i);                                   \
            d2[gmBase + 2 * dk3 + 3 * dk4] = make_float2(x13r, x13i);                                   \
            d2[gmBase + 2 * dk3 + 4 * dk4] = make_float2(x14r, x14i);                                   \
            d2[gmBase + 3 * dk3] = make_float2(x15r, x15i);                                             \
            d2[gmBase + 3 * dk3 + dk4] = make_float2(x16r, x16i);                                       \
            d2[gmBase + 3 * dk3 + 2 * dk4] = make_float2(x17r, x17i);                                   \
            d2[gmBase + 3 * dk3 + 3 * dk4] = make_float2(x18r, x18i);                                   \
            d2[gmBase + 3 * dk3 + 4 * dk4] = make_float2(x19r, x19i);                                   \
            d2[gmBase + 4 * dk3] = make_float2(x20r, x20i);                                             \
            d2[gmBase + 4 * dk3 + dk4] = make_float2(x21r, x21i);                                       \
            d2[gmBase + 4 * dk3 + 2 * dk4] = make_float2(x22r, x22i);                                   \
            d2[gmBase + 4 * dk3 + 3 * dk4] = make_float2(x23r, x23i);                                   \
            d2[gmBase + 4 * dk3 + 4 * dk4] = make_float2(x24r, x24i);                                   \
        }                                                                                               \
    } while (0)

namespace {
template <int32_t SGN>
__simt_vf__ __launch_bounds__(NF_SIMT_THREADS) inline void nfused_500_impl(__ubuf__ float* tile, __ubuf__ float* tw,
                                                                           __gm__ float* out, int32_t fftM)
{
    const float sgn = static_cast<float>(SGN);
    NF_STAGE20_UB(0, fftM, sgn);
    asc_syncthreads();
    NF_STAGE25_SCATTER(0, 0, fftM, sgn);
}
} // namespace

#undef NFD5
#undef NFR4
#undef NFTWMUL
#undef NFDIGREV20
#undef NF_STAGE20_UB
#undef NF_STAGE25_SCATTER

extern "C" __global__ __vector__ void fft_c2c_arch35_nfused_multi_core(__gm__ float* input, __gm__ float* twiddle,
                                                                       __gm__ float* output, __gm__ uint8_t* tiling)
{
    __gm__ NFusedTilingData* td = (__gm__ NFusedTilingData*)tiling;
    const int64_t batch = td->batch;
    const int64_t fftM = td->fftM;
    const int32_t direction = td->direction;
    const int64_t twFloatOff = td->twFloatOff;
    const int32_t coreNum = td->coreNum;
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= static_cast<uint32_t>(coreNum)) {
        return;
    }

    const int64_t per = batch / coreNum;
    const int64_t rem = batch % coreNum;
    const int64_t matStart = static_cast<int64_t>(blockIdx) * per + (blockIdx < rem ? blockIdx : rem);
    const int64_t matEnd = matStart + per + (blockIdx < rem ? 1 : 0);
    if (matStart >= matEnd) {
        return;
    }

    // UB 瓦片(与 example 同构):瓦片 [fftM,1000] 浮点 + 方向已选 twiddle 表 1040 浮点
    const int64_t matInFloats = fftM * NF_ROW_FLOATS;
    const int64_t matOutF2 = 500 * fftM;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tileBuf;
    TBuf<QuePosition::VECCALC> twBuf;
    pipe.InitBuffer(tileBuf, static_cast<uint32_t>(fftM) * NF_ROW_FLOATS * sizeof(float));
    pipe.InitBuffer(twBuf, 1040 * sizeof(float));
    LocalTensor<float> tile = tileBuf.Get<float>(static_cast<uint32_t>(fftM) * NF_ROW_FLOATS);
    LocalTensor<float> twUb = twBuf.Get<float>(1040);

    GlobalTensor<float> gmTw;
    gmTw.SetGlobalBuffer(twiddle + twFloatOff);
    DataCopy(twUb, gmTw, 1040);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    __ubuf__ float* twAddr = (__ubuf__ float*)twUb.GetPhyAddr();

    for (int64_t m = matStart; m < matEnd; ++m) {
        GlobalTensor<float> gmIn;
        gmIn.SetGlobalBuffer(input + m * matInFloats);
        DataCopy(tile, gmIn, static_cast<int32_t>(matInFloats));
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        AscendC::PipeBarrier<PIPE_V>();
        __ubuf__ float* tileAddr = (__ubuf__ float*)tile.GetPhyAddr();
        __gm__ float* out = output + m * matOutF2 * 2;
        if (direction == 1) {
            asc_vf_call<nfused_500_impl<1>>(dim3(NF_SIMT_THREADS), tileAddr, twAddr, out, static_cast<int>(fftM));
        } else {
            asc_vf_call<nfused_500_impl<-1>>(dim3(NF_SIMT_THREADS), tileAddr, twAddr, out, static_cast<int>(fftM));
        }
        AscendC::DataSyncBarrier<MemDsbT::UB>();
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}
