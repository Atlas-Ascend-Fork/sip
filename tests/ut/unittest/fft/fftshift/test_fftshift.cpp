/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "test_common.h"
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

namespace {

std::string GetFftShiftOutputDirectory()
{
    const char* currentDirEnv = std::getenv("CURRENT_DIR");
    if (currentDirEnv != nullptr) {
        return std::string(currentDirEnv);
    }
    return std::string(".");
}

// 确定性 LCG 填充：与 fftshift_data/gen_data.py 的输入流位级同源，
// 保证 numpy golden 与算子输入来自同一份比特流（用例可复现）
void FillLcg(std::vector<std::complex<float>>& buf, uint32_t seed)
{
    uint32_t x = seed;
    float* data = reinterpret_cast<float*>(buf.data());
    for (size_t i = 0; i < buf.size() * 2; i++) {
        x = x * 1664525u + 1013904223u;
        data[i] = static_cast<float>(static_cast<int32_t>(x >> 8)) * (1.0f / 8388608.0f);
    }
}

// 失败现场保留：把原始输出落盘，供 compare_data.py 事后比对定位
bool SaveRawBin(const std::vector<std::complex<float>>& data, const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()) * sizeof(std::complex<float>));
    return !file.fail();
}

// 管道流式生成 numpy golden：LCG 输入经 stdin 喂给 gen_data.py，输入不落盘，
// 仅 golden bin 落盘（共享环境磁盘紧张，峰值占用恒为单个 golden 文件）
int GenerateGoldenViaPipe(const std::string& dataDir, size_t caseIdx, int64_t row, int64_t col, int64_t batch,
                          int64_t dist)
{
    const int64_t sliceElems = row * col;
    const int64_t distVal = (dist > 0) ? dist : sliceElems;
    const int64_t inElems = (batch - 1) * distVal + sliceElems;

    std::string genCmd = "cd " + ShellQuote(dataDir) + " && python3 gen_data.py " + std::to_string(caseIdx) + " " +
                         std::to_string(row) + " " + std::to_string(col) + " " + std::to_string(batch) + " " +
                         std::to_string(dist);
    FILE* pipe = popen(genCmd.c_str(), "w");
    if (pipe == nullptr) {
        return -1;
    }
    std::vector<std::complex<float>> hostIn(static_cast<size_t>(inElems));
    FillLcg(hostIn, 12345u + static_cast<uint32_t>(caseIdx));
    constexpr size_t CHUNK_ELEMS = 1024 * 1024; // 8MB
    size_t total = static_cast<size_t>(inElems);
    for (size_t base = 0; base < total; base += CHUNK_ELEMS) {
        size_t n = (total - base < CHUNK_ELEMS) ? (total - base) : CHUNK_ELEMS;
        if (fwrite(hostIn.data() + base, sizeof(std::complex<float>), n, pipe) != n) {
            (void)pclose(pipe);
            return -1;
        }
    }
    return pclose(pipe);
}

// 算子执行 + 与 numpy golden 的二进制一致校验（进程内完成，与其他 FFT UT 一致）
void RunFftShiftOpCase(const std::string& dataDir, size_t caseIdx, int64_t row, int64_t col, int64_t batch,
                       int64_t dist)
{
    const int64_t sliceElems = row * col;
    const int64_t distVal = (dist > 0) ? dist : sliceElems;
    const int64_t inElems = (batch - 1) * distVal + sliceElems;
    const int64_t outElems = batch * sliceElems;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);

    std::vector<std::complex<float>> inputHostData(static_cast<size_t>(inElems));
    FillLcg(inputHostData, 12345u + static_cast<uint32_t>(caseIdx));
    std::vector<std::complex<float>> outputHostData(static_cast<size_t>(outElems), std::complex<float>(0.0f, 0.0f));

    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    // 输入为扁平复数向量（slice 间按 dist 步距分布），输出为 {batch, row, col} 连续布局
    std::vector<int64_t> inShape = {inElems};
    std::vector<int64_t> outShape = {batch, row, col};
    auto ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    // H2D 已完成，尽早归还输入 host 缓冲（典型 shape 时 2.4GB）
    std::vector<std::complex<float>>().swap(inputHostData);
    ret = CreateAclTensor(outputHostData, outShape, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutput);
    ASSERT_EQ(ret, 0);

    AspbStatus opStatus = asdFftShift(row, col, batch, dist, aclInput, aclOutput, aclStream, nullptr);
    ASSERT_EQ(opStatus, AsdSip::ErrorType::ACL_SUCCESS);

    ret = aclrtSynchronizeStream(aclStream);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    ret = aclrtMemcpy(outputHostData.data(), static_cast<size_t>(outElems) * sizeof(std::complex<float>),
                      outputDeviceAddr, static_cast<size_t>(outElems) * sizeof(std::complex<float>),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    // Steady-state NPU execution time: relaunch the op with the SAME tensors
    // (no re-allocation, kernel-args cache issue does not apply).
    {
        constexpr int REPEAT = 10;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPEAT; i++) {
            (void)asdFftShift(row, col, batch, dist, aclInput, aclOutput, aclStream, nullptr);
        }
        (void)aclrtSynchronizeStream(aclStream);
        auto t1 = std::chrono::steady_clock::now();
        double avgMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPEAT;
        double bytesMoved = (static_cast<double>(inElems) + static_cast<double>(outElems)) *
                            static_cast<double>(sizeof(std::complex<float>));
        std::cout << "[TIMING] case " << caseIdx << " (" << row << "x" << col << "x" << batch << ", " << outElems
                  << " elems): op avg " << avgMs << " ms/call over " << REPEAT << " calls, " << (bytesMoved / 1e9)
                  << " GB moved/call, effective BW " << (bytesMoved / (avgMs * 1e-3) / 1e9) << " GB/s" << std::endl;
    }

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);

    // FftShift 是纯排列算子，无浮点误差：rtol = atol = 0 时公共接口
    // CompareGoldenWithOutput 即逐位一致校验；失败时保留输出 bin 供
    // compare_data.py 事后定位
    std::string goldenFile = dataDir + "/complex64_golden_" + std::to_string(caseIdx) + ".bin";
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    int res = CompareGoldenWithOutput(goldenFile, outputHostData.data(), static_cast<size_t>(outElems), true, 0.0, 0.0,
                                      &maxAbsErr, &maxRelErr);
    std::cout << "compare result = " << (res == 0 ? 0 : 1) << " (mismatch=" << res << ", max_abs=" << maxAbsErr
              << ", max_rel=" << maxRelErr << ", binary exact, " << outElems << " complex64 elements)" << std::endl;
    if (res != 0) {
        SaveRawBin(outputHostData, dataDir + "/complex64_output_" + std::to_string(caseIdx) + ".bin");
    }
    ASSERT_EQ(res, 0);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

/*
 * FftShift 为纯数据排列算子，UT 要求输出与 numpy golden（gen_data.py：
 * dst[s] = np.roll(src[s], ((row + 1) / 2, (col + 1) / 2))，与参考 CUDA
 * fftShiftBatch2DKernel 逐位一致）二进制相同，比对复用公共
 * CompareGoldenWithOutput（rtol = atol = 0）。每用例流程：LCG 输入经管道
 * 流式送入 gen_data.py 生成 golden（输入不落盘，磁盘峰值恒为单个 golden
 * 文件，典型 shape 约 2.4GB）→ 进程内执行算子并比对 → 用例结束释放 golden。
 */
void RunFftShiftTest(size_t caseIdx, int64_t row, int64_t col, int64_t batch, int64_t dist)
{
    // suite 级一次性目录准备（对齐其他 FFT UT，含 GTEST 分片隔离）
    static std::string destPath = PrepareDataDirOnce(GetFftShiftOutputDirectory(), "fft/fftshift", "fftshift_data");
    std::string dataDir = destPath + "/fftshift_data";

    std::cout << "[case " << caseIdx << "] row=" << row << " col=" << col << " batch=" << batch << " dist=" << dist
              << std::endl;

    // golden 生成失败必须失败用例，否则残留旧参数的 golden 会造成"假通过"
    int genRes = GenerateGoldenViaPipe(dataDir, caseIdx, row, col, batch, dist);
    ASSERT_EQ(genRes, 0);

    RunFftShiftOpCase(dataDir, caseIdx, row, col, batch, dist);

    // 释放本用例 golden（失败时在上方提前返回，保留现场），磁盘峰值与用例数无关
    std::string goldenFile = dataDir + "/complex64_golden_" + std::to_string(caseIdx) + ".bin";
    (void)std::remove(goldenFile.c_str());
}

} // namespace

// Even/odd dimension coverage (contiguous slices, dist = row * col)
TEST(TestFftShift, TestShiftEvenEven) { RunFftShiftTest(0, 6, 8, 3, 0); }
TEST(TestFftShift, TestShiftOddOdd) { RunFftShiftTest(1, 7, 9, 2, 0); }
TEST(TestFftShift, TestShiftSingleRow) { RunFftShiftTest(2, 1, 16, 4, 0); }
TEST(TestFftShift, TestShiftSingleColumn) { RunFftShiftTest(3, 16, 1, 2, 0); }

// Strided input slices (dist > row * col, inter-slice padding)
TEST(TestFftShift, TestShiftStridedSlices) { RunFftShiftTest(4, 4, 6, 3, 32); }

// Medium shapes (odd x odd, col % 8 != 0, col % 8 == 0)
TEST(TestFftShift, TestShiftOddMedium) { RunFftShiftTest(5, 33, 67, 5, 0); }
TEST(TestFftShift, TestShiftColumnNotAligned) { RunFftShiftTest(6, 128, 50, 6, 0); }
TEST(TestFftShift, TestShiftColumnAligned) { RunFftShiftTest(7, 64, 48, 8, 0); }

// Typical MRI shape: RO x CH x PE x SPE = 500 x 24 x 500 x 50, one full un-chunked
// call with batch = RO * CH = 12000, row = PE = 500, col = SPE = 50, contiguous
// slices; 300M complex elements (2.4GB) per tensor
TEST(TestFftShift, TestShiftTypicalMriShape) { RunFftShiftTest(8, 500, 50, 12000, 0); }

// Very wide columns: a whole row cannot be staged per UB slot, so the kernel
// takes the piecewise fallback with per-row tile-sized sub-blocks (segLen >
// staging tile). shiftC = 5000 > 4096 exercises the wide-column split.
TEST(TestFftShift, TestShiftVeryWideColumn) { RunFftShiftTest(9, 2, 10000, 2, 0); }
