/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 性能测试驱动(独立程序,不依赖 gtest):
//   perf_fft_driver 1d <nfft> <batch> <iters> [vertical]
//   perf_fft_driver 2d <n1> <n2> <batch> <iters>
// 流程:初始化 → 建张量(固定种子随机填充)→ 建计划 → 预热 5 次 →
//       正式 iters 次(每次 exec+sync 记墙钟)→ 输出 min/median/max。
// 环境变量:PERF_DEVICE=卡号(默认 0);PERF_DUMP=输出bin路径(单次执行并转存,
//           供离线 numpy 精度门禁,默认关闭)。
// 正式耗时口径以 msprof 任务表为准,本程序墙钟仅作对照。

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "fft_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;
using namespace Mki;

static void PrintUsage()
{
    printf("usage:\n");
    printf("  perf_fft_driver 1d <nfft> <batch> <iters> [vertical]\n");
    printf("  perf_fft_driver 2d <n1> <n2> <batch> <iters>\n");
    printf("  perf_fft_driver 2drt <n1> <n2> <batch> <rounds>\n");
    printf("  perf_fft_driver 2drtf <n1> <n2> <batch> <rounds>\n");
}

static void PrintStats(const std::string& tag, const std::vector<double>& v)
{
    if (v.empty()) {
        return;
    }
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    double med = (s.size() % 2 == 1) ? s[s.size() / 2] : (s[s.size() / 2 - 1] + s[s.size() / 2]) / 2.0;
    printf("[perf] %s min=%.3f median=%.3f max=%.3f mean=%.3f (ms, n=%zu)\n", tag.c_str(), s.front(), med, s.back(),
           sum / s.size(), s.size());
    printf("[perf] %s samples:", tag.c_str());
    for (double x : v) {
        printf(" %.3f", x);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    std::string mode = (argc >= 2) ? argv[1] : "";

    // 批量扫描模式:一次进程按形状表顺序跑完全部形状
    // 用法:perf_fft_driver sweep <shapes_file> <iters>
    // shapes_file 每行:1d <nfft> <batch>(# 开头为注释)
    // 每个形状:打印宿主毫秒边界(供 msprof 任务表按时间窗归属)→ 建计划 →
    //           预热 5 次 → 正式 iters 次(逐次墙钟)→ 三点统计
    if (mode == "sweep") {
        if (argc < 4) {
            PrintUsage();
            return 1;
        }
        std::ifstream fin(argv[2]);
        if (!fin) {
            printf("[sweep] cannot open shapes file: %s\n", argv[2]);
            return 1;
        }
        int64_t sweepIters = atoll(argv[3]);
        struct SweepShape {
            bool is2d;
            int64_t n1;
            int64_t n2;
            int64_t batch;
            bool inv;
        };
        std::vector<SweepShape> shapes;
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream ss(line);
            std::string kind;
            int64_t a = 0, b = 0, c = 0;
            ss >> kind >> a >> b >> c;
            std::string dirTok;
            ss >> dirTok;
            bool inv = (dirTok == "inv" || dirTok == "inverse");
            if (kind == "1d" && a > 0 && b > 0) {
                shapes.push_back({false, a, 0, b, inv});
            } else if (kind == "2d" && a > 0 && b > 0 && c > 0) {
                shapes.push_back({true, a, b, c, inv});
            }
        }
        printf("[sweep] shapes=%zu iters=%ld\n", shapes.size(), (long)sweepIters);

        int deviceId = 0;
        const char* sweepDev = std::getenv("PERF_DEVICE");
        if (sweepDev != nullptr) {
            deviceId = atoi(sweepDev);
        }
        MkiRtStream stream = OpTestInit(deviceId);
        aclrtStream aclStream = static_cast<aclrtStream>(stream);
        printf("[sweep] device=%d\n", deviceId);
        auto epochMs = []() {
            return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        };

        for (size_t s = 0; s < shapes.size(); s++) {
            int64_t n1S = shapes[s].n1;
            int64_t n2S = shapes[s].n2;
            int64_t batchS = shapes[s].batch;
            int64_t nfftS = shapes[s].is2d ? n1S * n2S : n1S;
            printf("[sweep] begin shape=%zu nfft=%ld batch=%ld host_ms=%lld\n", s, (long)nfftS, (long)batchS,
                   epochMs());

            std::minstd_rand gen(9527);
            std::uniform_real_distribution<float> dis(-300.0f, 300.0f);
            std::vector<std::complex<float>> inHost(batchS * nfftS);
            for (auto& v : inHost) {
                v = std::complex<float>(dis(gen), dis(gen));
            }
            std::vector<std::complex<float>> outHost(batchS * nfftS, std::complex<float>(0.0f, 0.0f));
            std::vector<int64_t> shape = shapes[s].is2d ? std::vector<int64_t>{batchS, n1S, n2S} :
                                                          std::vector<int64_t>{batchS, n1S};

            aclTensor* aclIn = nullptr;
            aclTensor* aclOut = nullptr;
            void* inDev = nullptr;
            void* outDev = nullptr;
            if (CreateAclTensor(inHost, shape, &inDev, aclDataType::ACL_COMPLEX64, &aclIn) != 0) {
                printf("[sweep] CreateAclTensor in failed\n");
                return 1;
            }
            if (CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_COMPLEX64, &aclOut) != 0) {
                printf("[sweep] CreateAclTensor out failed\n");
                return 1;
            }

            asdFftHandle handle;
            if (asdFftCreate(handle) != AsdSip::ErrorType::ACL_SUCCESS) {
                printf("[sweep] asdFftCreate failed\n");
                return 1;
            }
            asdFftDirection sweepDir = shapes[s].inv ? asdFftDirection::ASCEND_FFT_INVERSE :
                                                       asdFftDirection::ASCEND_FFT_FORWARD;
            AspbStatus planStatus;
            if (shapes[s].is2d) {
                planStatus = asdFftMakePlan2D(handle, n1S, n2S, asdFftType::ASCEND_FFT_C2C, sweepDir,
                                              static_cast<int32_t>(batchS));
            } else {
                planStatus = asdFftMakePlan1D(handle, n1S, asdFftType::ASCEND_FFT_C2C, sweepDir,
                                              static_cast<int32_t>(batchS));
            }
            if (planStatus != AsdSip::ErrorType::ACL_SUCCESS) {
                printf("[sweep] make plan failed shape=%zu\n", s);
                return 1;
            }
            size_t workSize = 0;
            asdFftGetWorkspaceSize(handle, workSize);
            void* ws = nullptr;
            if (workSize > 0) {
                if (aclrtMalloc(&ws, (int64_t)workSize, ACL_MEM_MALLOC_HUGE_FIRST) != ::ACL_SUCCESS) {
                    printf("[sweep] workspace malloc failed\n");
                    return 1;
                }
                asdFftSetWorkspace(handle, (uint8_t*)ws);
            }
            asdFftSetStream(handle, aclStream);

            for (int i = 0; i < 5; i++) {
                if (asdFftExecC2C(handle, aclIn, aclOut) != AsdSip::ErrorType::ACL_SUCCESS ||
                    asdFftSynchronize(handle) != AsdSip::ErrorType::ACL_SUCCESS) {
                    printf("[sweep] warmup failed shape=%zu\n", s);
                    return 1;
                }
            }
            std::vector<double> wallMs;
            for (int64_t i = 0; i < sweepIters; i++) {
                auto t0 = std::chrono::steady_clock::now();
                if (asdFftExecC2C(handle, aclIn, aclOut) != AsdSip::ErrorType::ACL_SUCCESS ||
                    asdFftSynchronize(handle) != AsdSip::ErrorType::ACL_SUCCESS) {
                    printf("[sweep] exec failed shape=%zu iter=%ld\n", s, (long)i);
                    return 1;
                }
                auto t1 = std::chrono::steady_clock::now();
                wallMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            printf("[sweep] end shape=%zu host_ms=%lld\n", s, epochMs());
            PrintStats(("sweep nfft=" + std::to_string(nfftS) + " batch=" + std::to_string(batchS)).c_str(), wallMs);

            asdFftDestroy(handle);
            aclDestroyTensor(aclIn);
            aclDestroyTensor(aclOut);
            aclrtFree(inDev);
            aclrtFree(outDev);
            if (ws != nullptr) {
                aclrtFree(ws);
            }
        }
        TensorContext sweepContext;
        OpTestEnd(deviceId, sweepContext, stream);
        return 0;
    }

    if (mode == "2drtf") {
        // 2drtf <n1> <n2> <batch> <rounds>: 完整客户链路(fftshift→fwd→inv→ifftshift×1/N)
        // 数据用 example 同款 LCG(seed 42,[-1,1)),精度门 1e-5 绝对(example 口径)
        if (argc < 6) {
            PrintUsage();
            return 1;
        }
        int64_t n1rt = atoll(argv[2]);
        int64_t n2rt = atoll(argv[3]);
        int64_t batchRt = atoll(argv[4]);
        int64_t rounds = atoll(argv[5]);
        int deviceId = 0;
        const char* sweepDev = std::getenv("PERF_DEVICE");
        if (sweepDev != nullptr) {
            deviceId = atoi(sweepDev);
        }
        MkiRtStream stream = OpTestInit(deviceId);
        aclrtStream aclStream = static_cast<aclrtStream>(stream);

        int64_t numel = batchRt * n1rt * n2rt;
        std::vector<std::complex<float>> inHost(numel);
        {
            unsigned int seed = 42;
            for (int64_t i = 0; i < numel; ++i) {
                seed = seed * 1103515245u + 12345u;
                float v = static_cast<float>(static_cast<int32_t>(seed >> 16) & 0xffff) / 32768.0f - 1.0f;
                inHost[i] = std::complex<float>(v, v);
            }
        }
        std::vector<int64_t> shape = {batchRt, n1rt, n2rt};

        aclTensor* aclA = nullptr;
        aclTensor* aclB = nullptr;
        aclTensor* aclC = nullptr;
        void* devA = nullptr;
        void* devB = nullptr;
        void* devC = nullptr;
        if (CreateAclTensor(inHost, shape, &devA, aclDataType::ACL_COMPLEX64, &aclA) != 0 ||
            CreateAclTensor(inHost, shape, &devB, aclDataType::ACL_COMPLEX64, &aclB) != 0 ||
            CreateAclTensor(inHost, shape, &devC, aclDataType::ACL_COMPLEX64, &aclC) != 0) {
            printf("[2drtf] CreateAclTensor failed\n");
            return 1;
        }

        asdFftHandle hF;
        asdFftHandle hI;
        if (asdFftCreate(hF) != AsdSip::ErrorType::ACL_SUCCESS || asdFftCreate(hI) != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[2drtf] asdFftCreate failed\n");
            return 1;
        }
        if (asdFftMakePlan2D(hF, n1rt, n2rt, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD,
                             static_cast<int32_t>(batchRt)) != AsdSip::ErrorType::ACL_SUCCESS ||
            asdFftMakePlan2D(hI, n1rt, n2rt, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_INVERSE,
                             static_cast<int32_t>(batchRt)) != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[2drtf] make plan failed\n");
            return 1;
        }
        size_t workF = 0;
        size_t workI = 0;
        asdFftGetWorkspaceSize(hF, workF);
        asdFftGetWorkspaceSize(hI, workI);
        void* wsF = nullptr;
        void* wsI = nullptr;
        if (workF > 0 && aclrtMalloc(&wsF, (int64_t)workF, ACL_MEM_MALLOC_HUGE_FIRST) != ::ACL_SUCCESS) {
            printf("[2drtf] workspace F malloc failed\n");
            return 1;
        }
        if (workI > 0 && aclrtMalloc(&wsI, (int64_t)workI, ACL_MEM_MALLOC_HUGE_FIRST) != ::ACL_SUCCESS) {
            printf("[2drtf] workspace I malloc failed\n");
            return 1;
        }
        if (workF > 0) {
            asdFftSetWorkspace(hF, (uint8_t*)wsF);
        }
        if (workI > 0) {
            asdFftSetWorkspace(hI, (uint8_t*)wsI);
        }
        asdFftSetStream(hF, aclStream);
        asdFftSetStream(hI, aclStream);

        const int64_t dist = n1rt * n2rt;
        const double invScale = 1.0 / static_cast<double>(dist);
        aclrtEvent eS0;
        aclrtEvent eS1;
        aclrtEvent eI0;
        aclrtEvent eI1;
        aclrtCreateEvent(&eS0);
        aclrtCreateEvent(&eS1);
        aclrtCreateEvent(&eI0);
        aclrtCreateEvent(&eI1);
        double sumShift = 0.0;
        double sumIshift = 0.0;
        printf("[2drtf] n1=%ld n2=%ld batch=%ld rounds=%ld device=%d invScale=%g\n", (long)n1rt, (long)n2rt,
               (long)batchRt, (long)rounds, deviceId, invScale);

        std::vector<double> roundMs;
        double totalMs = 0.0;
        for (int64_t r = 0; r < rounds; r++) {
            auto t0 = std::chrono::steady_clock::now();
            aclrtRecordEvent(eS0, aclStream);
            if (asdFftShift(n1rt, n2rt, batchRt, dist, aclA, aclB, aclStream) != AsdSip::ErrorType::ACL_SUCCESS) {
                printf("[2drtf] fftshift failed round=%ld\n", (long)r);
                return 1;
            }
            aclrtRecordEvent(eS1, aclStream);
            if (asdFftExecC2C(hF, aclB, aclC) != AsdSip::ErrorType::ACL_SUCCESS) {
                printf("[2drtf] fwd exec failed round=%ld\n", (long)r);
                return 1;
            }
            if (asdFftExecC2C(hI, aclC, aclB) != AsdSip::ErrorType::ACL_SUCCESS) {
                printf("[2drtf] inv exec failed round=%ld\n", (long)r);
                return 1;
            }
            aclrtRecordEvent(eI0, aclStream);
            if (asdIfftShift(n1rt, n2rt, batchRt, dist, invScale, aclB, aclA, aclStream) !=
                AsdSip::ErrorType::ACL_SUCCESS) {
                printf("[2drtf] ifftshift failed round=%ld\n", (long)r);
                return 1;
            }
            aclrtRecordEvent(eI1, aclStream);
            if (aclrtSynchronizeStream(aclStream) != ::ACL_SUCCESS) {
                printf("[2drtf] sync failed round=%ld\n", (long)r);
                return 1;
            }
            float devS = 0.0f;
            float devI = 0.0f;
            aclrtEventElapsedTime(&devS, eS0, eS1);
            aclrtEventElapsedTime(&devI, eI0, eI1);
            sumShift += devS;
            sumIshift += devI;
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            roundMs.push_back(ms);
            totalMs += ms;
        }
        PrintStats(("2drtf round n1=" + std::to_string(n1rt) + " n2=" + std::to_string(n2rt)).c_str(), roundMs);
        printf("[2drtf] TOTAL %ld rounds = %.3f ms\n", (long)rounds, totalMs);
        printf("[2drtf] device avg over %ld rounds: shift=%.3f ms ifftshift=%.3f ms\n", (long)rounds, sumShift / rounds,
               sumIshift / rounds);

        // 往返精度:A 应还原为原始输入(example 口径:maxDiff < 1e-5)
        {
            std::vector<std::complex<float>> back(numel);
            if (aclrtMemcpy(back.data(), numel * sizeof(std::complex<float>), devA, numel * sizeof(std::complex<float>),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ::ACL_SUCCESS) {
                printf("[2drtf] memcpy back failed\n");
                return 1;
            }
            double maxDiff = 0.0;
            for (int64_t i = 0; i < numel; ++i) {
                const double dr = std::fabs(static_cast<double>(back[i].real()) - inHost[i].real());
                const double di = std::fabs(static_cast<double>(back[i].imag()) - inHost[i].imag());
                if (dr > maxDiff) {
                    maxDiff = dr;
                }
                if (di > maxDiff) {
                    maxDiff = di;
                }
            }
            printf("[2drtf] roundtrip maxDiff=%.3e (gate 1e-5) verdict=%s\n", maxDiff,
                   maxDiff < 1e-5 ? "PASSED" : "FAILED");
        }

        aclrtDestroyEvent(eS0);
        aclrtDestroyEvent(eS1);
        aclrtDestroyEvent(eI0);
        aclrtDestroyEvent(eI1);
        aclDestroyTensor(aclA);
        aclDestroyTensor(aclB);
        aclDestroyTensor(aclC);
        aclrtFree(devA);
        aclrtFree(devB);
        aclrtFree(devC);
        if (wsF != nullptr) {
            aclrtFree(wsF);
        }
        if (wsI != nullptr) {
            aclrtFree(wsI);
        }
        asdFftDestroy(hF);
        asdFftDestroy(hI);
        TensorContext rtContext;
        OpTestEnd(deviceId, rtContext, stream);
        return 0;
    }

    if (mode == "2drt") {
        // 2drt <n1> <n2> <batch> <rounds>: 每轮 fft2dC2C + ifft2dC2C 各一次,按轮计时
        if (argc < 6) {
            PrintUsage();
            return 1;
        }
        int64_t n1rt = atoll(argv[2]);
        int64_t n2rt = atoll(argv[3]);
        int64_t batchRt = atoll(argv[4]);
        int64_t rounds = atoll(argv[5]);
        int deviceId = 0;
        const char* sweepDev = std::getenv("PERF_DEVICE");
        if (sweepDev != nullptr) {
            deviceId = atoi(sweepDev);
        }
        MkiRtStream stream = OpTestInit(deviceId);
        aclrtStream aclStream = static_cast<aclrtStream>(stream);

        int64_t numel = batchRt * n1rt * n2rt;
        std::vector<std::complex<float>> inHost(numel);
        std::minstd_rand gen(9527);
        std::uniform_real_distribution<float> dis(-300.0f, 300.0f);
        for (auto& v : inHost) {
            v = std::complex<float>(dis(gen), dis(gen));
        }
        std::vector<std::complex<float>> outHost(numel, std::complex<float>(0.0f, 0.0f));
        std::vector<int64_t> shape = {batchRt, n1rt, n2rt};

        aclTensor* aclA = nullptr;
        aclTensor* aclB = nullptr;
        void* devA = nullptr;
        void* devB = nullptr;
        if (CreateAclTensor(inHost, shape, &devA, aclDataType::ACL_COMPLEX64, &aclA) != 0) {
            printf("[2drt] CreateAclTensor A failed\n");
            return 1;
        }
        if (CreateAclTensor(outHost, shape, &devB, aclDataType::ACL_COMPLEX64, &aclB) != 0) {
            printf("[2drt] CreateAclTensor B failed\n");
            return 1;
        }

        asdFftHandle hF;
        asdFftHandle hI;
        if (asdFftCreate(hF) != AsdSip::ErrorType::ACL_SUCCESS || asdFftCreate(hI) != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[2drt] asdFftCreate failed\n");
            return 1;
        }
        if (asdFftMakePlan2D(hF, n1rt, n2rt, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD,
                             static_cast<int32_t>(batchRt)) != AsdSip::ErrorType::ACL_SUCCESS ||
            asdFftMakePlan2D(hI, n1rt, n2rt, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_INVERSE,
                             static_cast<int32_t>(batchRt)) != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[2drt] make plan failed\n");
            return 1;
        }
        size_t workF = 0;
        size_t workI = 0;
        asdFftGetWorkspaceSize(hF, workF);
        asdFftGetWorkspaceSize(hI, workI);
        void* wsF = nullptr;
        void* wsI = nullptr;
        if (workF > 0 && aclrtMalloc(&wsF, (int64_t)workF, ACL_MEM_MALLOC_HUGE_FIRST) != ::ACL_SUCCESS) {
            printf("[2drt] workspace F malloc failed\n");
            return 1;
        }
        if (workI > 0 && aclrtMalloc(&wsI, (int64_t)workI, ACL_MEM_MALLOC_HUGE_FIRST) != ::ACL_SUCCESS) {
            printf("[2drt] workspace I malloc failed\n");
            return 1;
        }
        if (workF > 0) {
            asdFftSetWorkspace(hF, (uint8_t*)wsF);
        }
        if (workI > 0) {
            asdFftSetWorkspace(hI, (uint8_t*)wsI);
        }
        asdFftSetStream(hF, aclStream);
        asdFftSetStream(hI, aclStream);
        aclrtEvent evtM0;
        aclrtEvent evtM1;
        aclrtEvent evtN0;
        aclrtEvent evtN1;
        aclrtCreateEvent(&evtM0);
        aclrtCreateEvent(&evtM1);
        aclrtCreateEvent(&evtN0);
        aclrtCreateEvent(&evtN1);
        double sumM = 0.0;
        double sumN = 0.0;
        double sumGap = 0.0;
        printf("[2drt] n1=%ld n2=%ld batch=%ld rounds=%ld device=%d\n", (long)n1rt, (long)n2rt, (long)batchRt,
               (long)rounds, deviceId);

        std::vector<double> roundMs;
        double totalMs = 0.0;
        for (int64_t r = 0; r < rounds; r++) {
            auto t0 = std::chrono::steady_clock::now();
            aclrtRecordEvent(evtM0, aclStream);
            if (asdFftExecC2C(hF, aclA, aclB) != AsdSip::ErrorType::ACL_SUCCESS) {
                printf("[2drt] fwd exec failed round=%ld\n", (long)r);
                return 1;
            }
            aclrtRecordEvent(evtM1, aclStream);
            aclrtRecordEvent(evtN0, aclStream);
            if (asdFftExecC2C(hI, aclB, aclA) != AsdSip::ErrorType::ACL_SUCCESS) {
                printf("[2drt] inv exec failed round=%ld\n", (long)r);
                return 1;
            }
            aclrtRecordEvent(evtN1, aclStream);
            if (aclrtSynchronizeStream(aclStream) != ::ACL_SUCCESS) {
                printf("[2drt] sync failed round=%ld\n", (long)r);
                return 1;
            }
            float devM = 0.0f;
            float devN = 0.0f;
            float devGap = 0.0f;
            aclrtEventElapsedTime(&devM, evtM0, evtM1);
            aclrtEventElapsedTime(&devN, evtN0, evtN1);
            aclrtEventElapsedTime(&devGap, evtM1, evtN0);
            sumM += devM;
            sumN += devN;
            sumGap += devGap;
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            roundMs.push_back(ms);
            totalMs += ms;
        }
        PrintStats(("2drt round n1=" + std::to_string(n1rt) + " n2=" + std::to_string(n2rt)).c_str(), roundMs);
        printf("[2drt] TOTAL %ld rounds = %.3f ms\n", (long)rounds, totalMs);
        printf("[2drt] device avg over %ld rounds: M=%.3f ms N=%.3f ms interPassGap=%.3f ms (M+N+gap=%.3f)\n",
               (long)rounds, sumM / rounds, sumN / rounds, sumGap / rounds, (sumM + sumN + sumGap) / rounds);
        aclrtDestroyEvent(evtM0);
        aclrtDestroyEvent(evtM1);
        aclrtDestroyEvent(evtN0);
        aclrtDestroyEvent(evtN1);

        aclDestroyTensor(aclA);
        aclDestroyTensor(aclB);
        aclrtFree(devA);
        aclrtFree(devB);
        if (wsF != nullptr) {
            aclrtFree(wsF);
        }
        if (wsI != nullptr) {
            aclrtFree(wsI);
        }
        asdFftDestroy(hF);
        asdFftDestroy(hI);
        TensorContext rtContext;
        OpTestEnd(deviceId, rtContext, stream);
        return 0;
    }

    if (argc < 5) {
        PrintUsage();
        return 1;
    }

    int64_t nfft = 0, batch = 0, n1 = 0, n2 = 0, iters = 20;
    bool vertical = false;
    if (mode == "1d") {
        nfft = atoll(argv[2]);
        batch = atoll(argv[3]);
        iters = atoll(argv[4]);
        vertical = (argc >= 6 && std::string(argv[5]) == "vertical");
        if (nfft <= 0 || batch <= 0) {
            PrintUsage();
            return 1;
        }
    } else if (mode == "2d") {
        if (argc < 6) {
            PrintUsage();
            return 1;
        }
        n1 = atoll(argv[2]);
        n2 = atoll(argv[3]);
        batch = atoll(argv[4]);
        iters = atoll(argv[5]);
        if (n1 <= 0 || n2 <= 0 || batch <= 0) {
            PrintUsage();
            return 1;
        }
    } else {
        PrintUsage();
        return 1;
    }

    int deviceId = 0;
    const char* devEnv = std::getenv("PERF_DEVICE");
    if (devEnv != nullptr) {
        deviceId = atoi(devEnv);
    }

    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    if (mode == "1d") {
        inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, nfft}});
        inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, nfft}});
    } else {
        inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, n1, n2}});
        inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, n1, n2}});
    }

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);
    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    int64_t numel = (mode == "1d") ? batch * nfft : batch * n1 * n2;
    std::vector<std::complex<float>> inputHostData(numel);
    std::complex<float>* srcPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (int64_t i = 0; i < numel; i++) {
        inputHostData[i] = srcPtr[i];
    }
    std::vector<std::complex<float>> outputHostData(numel, std::complex<float>(0.0f, 0.0f));

    std::vector<int64_t> shape;
    if (mode == "1d") {
        shape = {batch, nfft};
    } else {
        shape = {batch, n1, n2};
    }

    auto ret = CreateAclTensor(inputHostData, shape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    if (ret != 0) {
        printf("[perf] CreateAclTensor input failed: %d\n", ret);
        return 1;
    }
    ret = CreateAclTensor(outputHostData, shape, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutput);
    if (ret != 0) {
        printf("[perf] CreateAclTensor output failed: %d\n", ret);
        return 1;
    }

    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        printf("[perf] asdFftCreate failed\n");
        return 1;
    }

    if (mode == "1d") {
        auto dimType = vertical ? asdFft1dDimType::ASCEND_FFT_VERTICAL : asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
        fftStatus = asdFftMakePlan1D(handle, nfft, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD,
                                     batch, dimType);
    } else {
        fftStatus = asdFftMakePlan2D(handle, n1, n2, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD,
                                     batch);
    }
    if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        printf("[perf] make plan failed\n");
        return 1;
    }

    size_t workSize = 0;
    fftStatus = asdFftGetWorkspaceSize(handle, workSize);
    if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        printf("[perf] get workspace size failed\n");
        return 1;
    }
    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ::ACL_SUCCESS) {
            printf("[perf] workspace malloc failed\n");
            return 1;
        }
        fftStatus = asdFftSetWorkspace(handle, (uint8_t*)workspaceAddr);
        if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[perf] set workspace failed\n");
            return 1;
        }
    }

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    fftStatus = asdFftSetStream(handle, aclStream);
    if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        printf("[perf] set stream failed\n");
        return 1;
    }

    printf("[perf] mode=%s nfft=%ld batch=%ld iters=%ld vertical=%d workspace=%zu device=%d\n", mode.c_str(),
           (long)(mode == "1d" ? nfft : n1 * n2), (long)batch, (long)iters, (int)vertical, workSize, deviceId);

    // 可选:单次执行并转存输出(离线精度门禁用),完成后直接退出
    const char* dumpEnv = std::getenv("PERF_DUMP");
    if (dumpEnv != nullptr) {
        fftStatus = asdFftExecC2C(handle, aclInput, aclOutput);
        if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[perf] dump exec failed\n");
            return 1;
        }
        fftStatus = asdFftSynchronize(handle);
        if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[perf] dump sync failed\n");
            return 1;
        }
        ret = aclrtMemcpy(outputTensor.hostData, outputTensor.dataSize, outputDeviceAddr, outputTensor.dataSize,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ::ACL_SUCCESS) {
            printf("[perf] dump d2h failed\n");
            return 1;
        }
        if (!SaveTensorToBin(outputTensor, std::string(dumpEnv) + ".output.bin")) {
            printf("[perf] dump save output failed: %s\n", dumpEnv);
            return 1;
        }
        if (!SaveTensorToBin(inputTensor, std::string(dumpEnv) + ".input.bin")) {
            printf("[perf] dump save input failed: %s\n", dumpEnv);
            return 1;
        }
        printf("[perf] dump saved: %s.output.bin / %s.input.bin\n", dumpEnv, dumpEnv);
        aclDestroyTensor(aclInput);
        aclDestroyTensor(aclOutput);
        aclrtFree(inputDeviceAddr);
        aclrtFree(outputDeviceAddr);
        if (workSize > 0) {
            aclrtFree(workspaceAddr);
        }
        asdFftDestroy(handle);
        OpTestEnd(deviceId, context, stream);
        return 0;
    }

    // 预热 5 次
    for (int i = 0; i < 5; i++) {
        fftStatus = asdFftExecC2C(handle, aclInput, aclOutput);
        if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[perf] warmup exec failed\n");
            return 1;
        }
        fftStatus = asdFftSynchronize(handle);
        if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
            const char* retryEnv = std::getenv("PERF_RETRY");
            if (i == 0 && retryEnv != nullptr) {
                printf("[perf] warmup sync failed (i=%d), RETRY exec+sync\n", i);
                fftStatus = asdFftExecC2C(handle, aclInput, aclOutput);
                printf("[perf] retry exec rc=%d\n", (int)fftStatus);
                if (fftStatus == AsdSip::ErrorType::ACL_SUCCESS) {
                    fftStatus = asdFftSynchronize(handle);
                    printf("[perf] retry sync rc=%d\n", (int)fftStatus);
                }
                if (fftStatus == AsdSip::ErrorType::ACL_SUCCESS) {
                    printf("[perf] RETRY_RECOVERED\n");
                    continue;
                }
                printf("[perf] RETRY_FAILED_TOO\n");
                return 1;
            }
            printf("[perf] warmup sync failed\n");
            return 1;
        }
    }

    // 正式 iters 次,逐次记墙钟
    std::vector<double> wallMs;
    for (int64_t i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        fftStatus = asdFftExecC2C(handle, aclInput, aclOutput);
        if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[perf] exec failed at iter %ld\n", (long)i);
            return 1;
        }
        fftStatus = asdFftSynchronize(handle);
        if (fftStatus != AsdSip::ErrorType::ACL_SUCCESS) {
            printf("[perf] sync failed at iter %ld\n", (long)i);
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        wallMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    PrintStats("wall", wallMs);

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }
    asdFftDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    return 0;
}
