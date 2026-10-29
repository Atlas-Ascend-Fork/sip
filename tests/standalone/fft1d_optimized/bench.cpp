/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Formal API measurement only; FFT implementations are selected by the loaded SIP library.
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "fft_api.h"

namespace {
constexpr int kDefaultDevice = 2;
void Check(int code, const char* call)
{
    if (code)
        throw std::runtime_error(std::string(call) + " code=" + std::to_string(code));
}
#define CHECK(call) Check(static_cast<int>(call), #call)

bool Nonempty(const char* name)
{
    const char* value = std::getenv(name);
    return value && value[0] != '\0';
}

int Device(const char* text)
{
    size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != std::strlen(text) || value < 0 || value > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("device must be a nonnegative int32 ACL device ID");
    return static_cast<int>(value);
}

bool SameFile(const char* first, const char* second)
{
    namespace fs = std::filesystem;
    return fs::weakly_canonical(first) == fs::weakly_canonical(second) ||
           (fs::exists(first) && fs::exists(second) && fs::equivalent(first, second));
}

size_t Bytes(const std::vector<int64_t>& shape, bool complex)
{
    size_t bytes = complex ? 2 * sizeof(float) : sizeof(float);
    for (const int64_t dimension : shape) {
        if (dimension < 1 || static_cast<uint64_t>(dimension) >
                                 static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / bytes)
            throw std::runtime_error("tensor byte count overflow");
        bytes *= dimension;
    }
    return bytes;
}

aclTensor* Tensor(void* data, const std::vector<int64_t>& shape, bool complex)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size() - 1; i > 0; --i)
        strides[i - 1] = strides[i] * shape[i];
    auto* tensor = aclCreateTensor(shape.data(), shape.size(), complex ? ACL_COMPLEX64 : ACL_FLOAT, strides.data(), 0,
                                   ACL_FORMAT_ND, shape.data(), shape.size(), data);
    if (!tensor)
        throw std::runtime_error("aclCreateTensor returned null");
    return tensor;
}
} // namespace

int main(int argc, char** argv)
{
    // Optional width/device preserve the original square/1D argument order.
    if (argc < 8 || argc > 14) {
        std::cerr << "Usage: fft_bench 1|2|3 c2c|r2c|c2r N batch input.bin output.bin timings.csv "
                     "[samples=300] [warmup=10] [standard|forward|inverse] [width=N] [device=2] [alias=0]\n";
        return 2;
    }
    try {
        const int dimension = std::stoi(argv[1]);
        const std::string op = argv[2];
        const int64_t n = std::stoll(argv[3]), batch = std::stoll(argv[4]);
        const int samples = argc >= 9 ? std::stoi(argv[8]) : 300;
        const int warmup = argc >= 10 ? std::stoi(argv[9]) : 10;
        const std::string direction = argc >= 11 ? argv[10] : "standard";
        const int64_t width = argc >= 12 ? std::stoll(argv[11]) : n;
        const int device = argc >= 13 ? Device(argv[12]) : kDefaultDevice;
        if ((dimension < 1 || dimension > 3) || (op != "r2c" && op != "c2r" && op != "c2c") || n < 2 || n > 8192 ||
            width < 2 || width > 8192 || (dimension == 1 && width != n) || batch < 1 ||
            batch > std::numeric_limits<int32_t>::max() || samples < 1 || warmup < 0 ||
            (direction != "standard" && direction != "forward" && direction != "inverse"))
            throw std::runtime_error("invalid dimension/op/N/width/batch/samples/warmup/direction");
        if (Nonempty("ASCEND_RT_VISIBLE_DEVICES") || Nonempty("ASCEND_VISIBLE_DEVICES") ||
            Nonempty("NPU_VISIBLE_DEVICES"))
            throw std::runtime_error("device remapping is forbidden; use the verified ACL device ID");
        if (SameFile(argv[5], argv[6]) || SameFile(argv[5], argv[7]) || SameFile(argv[6], argv[7]))
            throw std::runtime_error("input, output and timing paths must be different files");
        if (setenv("SIP_FFT_BACKEND", "internal", 1) || setenv("ASDSIP_FFT_PLAN_TRACE", "1", 1))
            throw std::runtime_error("cannot set internal backend/trace environment");
        const bool c2c = op == "c2c";
        const bool alias = argc == 14 && std::stoi(argv[13]) != 0;
        const bool c2r = op == "c2r";
        const bool forward = direction == "standard" ? !c2r : direction == "forward";
        std::vector<int64_t> realShape = dimension == 1 ? std::vector<int64_t>{batch, n} :
                                                          std::vector<int64_t>{batch, n, width};
        if (dimension == 3)
            realShape.push_back(width);
        std::vector<int64_t> halfShape = realShape;
        halfShape.back() = width / 2 + 1;
        const auto& inputShape = c2r ? halfShape : realShape;
        const auto& outputShape = (c2r || c2c) ? realShape : halfShape;
        const size_t inputBytes = Bytes(inputShape, c2r || c2c), outputBytes = Bytes(outputShape, !c2r);
        std::vector<char> input(inputBytes), output(outputBytes);
        std::ifstream source(argv[5], std::ios::binary);
        if (!source.read(input.data(), inputBytes) || source.peek() != EOF)
            throw std::runtime_error("input byte count/read mismatch");
        CHECK(aclInit(nullptr));
        CHECK(aclrtSetDevice(device));
        aclrtStream stream = nullptr;
        CHECK(aclrtCreateStream(&stream));
        void *devInput = nullptr, *devOutput = nullptr, *workspace = nullptr;
        CHECK(aclrtMalloc(&devInput, std::max(inputBytes, outputBytes), ACL_MEM_MALLOC_HUGE_FIRST));
        if (alias)
            devOutput = devInput;
        else
            CHECK(aclrtMalloc(&devOutput, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK(aclrtMemcpy(devInput, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        auto* inTensor = Tensor(devInput, inputShape, c2r || c2c);
        auto* outTensor = Tensor(devOutput, outputShape, !c2r);
        AsdSip::asdFftHandle handle = nullptr;
        CHECK(AsdSip::asdFftCreate(handle));
        CHECK(AsdSip::asdFftSetStream(handle, stream));
        const auto type = c2c ? AsdSip::ASCEND_FFT_C2C : c2r ? AsdSip::ASCEND_FFT_C2R : AsdSip::ASCEND_FFT_R2C;
        const auto sign = forward ? AsdSip::ASCEND_FFT_FORWARD : AsdSip::ASCEND_FFT_INVERSE;
        if (dimension == 1)
            CHECK(AsdSip::asdFftMakePlan1D(handle, n, type, sign, batch));
        else if (dimension == 3)
            CHECK(AsdSip::asdFftMakePlan3D(handle, n, width, width, type, sign, static_cast<int32_t>(batch)));
        else
            CHECK(AsdSip::asdFftMakePlan2D(handle, n, width, type, sign, static_cast<int32_t>(batch)));
        size_t workspaceBytes = 0;
        CHECK(AsdSip::asdFftGetWorkspaceSize(handle, workspaceBytes));
        if (workspaceBytes) {
            CHECK(aclrtMalloc(&workspace, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
            CHECK(AsdSip::asdFftSetWorkspace(handle, workspace));
        }
        auto execute = [&]() {
            if (c2c)
                CHECK(AsdSip::asdFftExecC2C(handle, inTensor, outTensor));
            else if (c2r)
                CHECK(AsdSip::asdFftExecC2R(handle, inTensor, outTensor));
            else
                CHECK(AsdSip::asdFftExecR2C(handle, inTensor, outTensor));
        };
        for (int i = 0; i < warmup; ++i) {
            if (alias)
                CHECK(aclrtMemcpy(devInput, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
            execute();
            CHECK(aclrtSynchronizeStream(stream));
        }
        aclrtEvent start = nullptr, end = nullptr;
        CHECK(aclrtCreateEvent(&start));
        CHECK(aclrtCreateEvent(&end));
        std::ofstream timing(argv[7]);
        timing << std::setprecision(15) << "sample,acl_us,host_sync_us\n";
        for (int i = 0; i < samples; ++i) {
            if (alias)
                CHECK(aclrtMemcpy(devInput, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
            const auto before = std::chrono::steady_clock::now();
            CHECK(aclrtRecordEvent(start, stream));
            execute();
            CHECK(aclrtRecordEvent(end, stream));
            CHECK(aclrtSynchronizeEvent(end));
            const auto after = std::chrono::steady_clock::now();
            float ms = 0;
            CHECK(aclrtEventElapsedTime(&ms, start, end));
            timing << i << ',' << static_cast<double>(ms) * 1000.0 << ','
                   << std::chrono::duration<double, std::micro>(after - before).count() << '\n';
        }
        CHECK(aclrtMemcpy(output.data(), outputBytes, devOutput, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream target(argv[6], std::ios::binary);
        target.write(output.data(), outputBytes);
        target.flush();
        timing.flush();
        if (!target || !timing)
            throw std::runtime_error("output/timing write failed");
        output.resize(inputBytes);
        CHECK(aclrtMemcpy(output.data(), inputBytes, devInput, inputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        const bool unchanged = alias || std::memcmp(input.data(), output.data(), inputBytes) == 0;
        std::cout << "RESULT dim=" << dimension << " op=" << op << " N=" << n << " M=" << width << " batch=" << batch
                  << " direction=" << (forward ? "forward" : "inverse") << " device=" << device << " normalization=none"
                  << " input_bytes=" << inputBytes << " output_bytes=" << outputBytes
                  << " workspace_bytes=" << workspaceBytes << " samples=" << samples << " warmup=" << warmup
                  << " alias=" << alias << " profiling=0 input_unchanged=" << unchanged << '\n';
        CHECK(AsdSip::asdFftDestroy(handle));
        CHECK(aclDestroyTensor(inTensor));
        CHECK(aclDestroyTensor(outTensor));
        CHECK(aclrtDestroyEvent(start));
        CHECK(aclrtDestroyEvent(end));
        CHECK(aclrtFree(devInput));
        if (!alias)
            CHECK(aclrtFree(devOutput));
        if (workspace)
            CHECK(aclrtFree(workspace));
        CHECK(aclrtDestroyStream(stream));
        // Shared card: release our resources without resetting the device.
        CHECK(aclFinalize());
        if (!unchanged)
            throw std::runtime_error("FFT execution modified its input");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FFT_BENCH_FAILURE: " << error.what() << '\n';
        return 1;
    }
}
