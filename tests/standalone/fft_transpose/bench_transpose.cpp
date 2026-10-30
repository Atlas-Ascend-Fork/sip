/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Validate the Transpose component used by FFT plans.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <acl/acl.h>
#include "fftoperation/transpose.h"

void Check(int code)
{
    if (code)
        throw std::runtime_error("ACL error " + std::to_string(code));
}

bool IsGuardWord(uint32_t value) { return value == 0xa5a5a5a5U; }

template <typename Function>
void ExpectInvalidBuffer(Function function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid transpose buffer was accepted");
}

void CheckInvalidBuffers(Transpose& op, void* input, void* output, size_t bytes, const Mki::SVector<int64_t>& dims,
                         aclrtStream stream, workspace::Workspace& workspace)
{
    Tensor in, out;
    in.desc = {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, dims, {}, 0};
    in.data = input;
    in.dataSize = bytes;
    out.data = output;
    out.dataSize = bytes - 8;
    ExpectInvalidBuffer([&]() { op.Run(in, out, stream, workspace); });
    out.dataSize = bytes;
    in.dataSize = bytes - 8;
    ExpectInvalidBuffer([&]() { op.Run(in, out, stream, workspace); });
    ExpectInvalidBuffer([&]() { op.Run(input, static_cast<uint8_t*>(input) + 8, stream, workspace); });
    ExpectInvalidBuffer([&]() { op.Run(input, nullptr, stream, workspace); });
    std::vector<uint32_t> untouched(bytes / sizeof(uint32_t));
    Check(aclrtMemcpy(untouched.data(), bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!std::all_of(untouched.begin(), untouched.end(), IsGuardWord)) {
        throw std::runtime_error("rejected buffer call modified output");
    }
}

struct TestResources {
    bool initialized = false;
    aclrtStream stream = nullptr;
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    void* input = nullptr;
    void* output = nullptr;
    void* scratch = nullptr;
    std::unique_ptr<Transpose> op;

    ~TestResources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        op.reset();
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (end != nullptr) {
            aclrtDestroyEvent(end);
        }
        if (scratch != nullptr) {
            aclrtFree(scratch);
        }
        if (input != nullptr) {
            aclrtFree(input);
        }
        if (output != nullptr) {
            aclrtFree(output);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

int main(int argc, char** argv)
{
    if (argc != 7 && argc != 8) {
        std::cerr << "Usage: transpose_bench rows cols batch samples device timings.csv "
                     "[raw|tensor|strides|axes01|capacity]"
                  << '\n';
        return 2;
    }
    try {
        const int64_t h = std::stoll(argv[1]), w = std::stoll(argv[2]), b = std::stoll(argv[3]);
        const int samples = std::stoi(argv[4]), device = std::stoi(argv[5]);
        const std::string mode = argc == 8 ? argv[7] : "raw";
        if (mode != "raw" && mode != "tensor" && mode != "strides" && mode != "axes01" && mode != "capacity")
            throw std::invalid_argument("invalid execution mode");
        if (h <= 0 || h > 32768 || w <= 0 || w > 32768 || b <= 0 || b > 8 || samples < 1 || device < 0 ||
            b * h * w > 8LL * 8192 * 8192)
            throw std::invalid_argument("invalid test shape/device");
        const size_t elements = static_cast<size_t>(b * h * w);
        const size_t bytes = elements * 8;
        constexpr size_t guard = 256;
        std::vector<uint32_t> x(elements * 2), y(elements * 2 + guard / 4);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = static_cast<uint32_t>(i * 2654435761U + 0x7f800000U);
        TestResources resources;
        Check(aclInit(nullptr));
        resources.initialized = true;
        Check(aclrtSetDevice(device));
        auto& stream = resources.stream;
        Check(aclrtCreateStream(&stream));
        auto& input = resources.input;
        auto& output = resources.output;
        auto& scratch = resources.scratch;
        Check(aclrtMalloc(&input, bytes + guard, ACL_MEM_MALLOC_HUGE_FIRST));
        Check(aclrtMalloc(&output, bytes + guard, ACL_MEM_MALLOC_HUGE_FIRST));
        Check(aclrtMemset(input, bytes + guard, 0xa5, bytes + guard));
        Check(aclrtMemset(output, bytes + guard, 0xa5, bytes + guard));
        Check(aclrtMemcpy(input, bytes, x.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        auto& op = resources.op;
        op = std::make_unique<Transpose>(mode == "axes01" ? 0 : -1, mode == "axes01" ? 1 : -2,
                                         Mki::SVector<int64_t>{b, h, w});
        if (!op->init())
            throw std::runtime_error("transpose init failed");
        const size_t workspaceBytes = op->EstimateWorkspaceSize();
        Check(aclrtMalloc(&scratch, std::max<size_t>(32, workspaceBytes), ACL_MEM_MALLOC_HUGE_FIRST));
        workspace::Workspace workspace(scratch, workspaceBytes);
        if (mode == "capacity") {
            if (elements <= 1) {
                throw std::invalid_argument("capacity checks require at least two complex elements");
            }
            CheckInvalidBuffers(*op, input, output, bytes, {b, h, w}, stream, workspace);
        }
        auto execute = [&]() {
            if (mode == "tensor" || mode == "strides" || mode == "capacity") {
                Tensor in, out;
                in.desc = {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, {b, h, w}, {}, 0};
                if (mode == "strides")
                    in.desc.strides = {h * w, w, 1};
                in.data = input;
                in.dataSize = bytes;
                out.data = output;
                out.dataSize = mode == "tensor" ? 0 : bytes;
                op->Run(in, out, stream, workspace);
                if (out.desc.dims != Mki::SVector<int64_t>{b, w, h} || !out.IsContiguous() ||
                    out.desc.dtype != Mki::TENSOR_DTYPE_COMPLEX64)
                    throw std::runtime_error("incorrect output descriptor");
            } else {
                op->Run(input, output, stream, workspace);
            }
        };
        for (int i = 0; i < 10; ++i) {
            execute();
            Check(aclrtSynchronizeStream(stream));
        }
        auto& start = resources.start;
        auto& end = resources.end;
        Check(aclrtCreateEvent(&start));
        Check(aclrtCreateEvent(&end));
        std::ofstream times(argv[6]);
        times << "sample,acl_us" << '\n';
        for (int i = 0; i < samples; ++i) {
            Check(aclrtRecordEvent(start, stream));
            execute();
            Check(aclrtRecordEvent(end, stream));
            Check(aclrtSynchronizeEvent(end));
            float ms = 0;
            Check(aclrtEventElapsedTime(&ms, start, end));
            times << i << ',' << ms * 1000.0 << '\n';
        }
        times.close();
        if (!times)
            throw std::runtime_error("timing write failed");
        Check(aclrtMemcpy(y.data(), bytes + guard, output, bytes + guard, ACL_MEMCPY_DEVICE_TO_HOST));
        for (int64_t batch = 0; batch < b; ++batch) {
            for (int64_t row = 0; row < h; ++row) {
                for (int64_t col = 0; col < w; ++col) {
                    const auto src = 2 * (batch * h * w + row * w + col);
                    const auto dst = mode == "axes01" ? 2 * (row * b * w + batch * w + col) :
                                                        2 * (batch * h * w + col * h + row);
                    if (x[src] != y[dst] || x[src + 1] != y[dst + 1])
                        throw std::runtime_error("transpose bitwise mismatch");
                }
            }
        }
        if (!std::all_of(y.begin() + elements * 2, y.end(), IsGuardWord))
            throw std::runtime_error("output guard modified");
        Check(aclrtMemcpy(y.data(), bytes + guard, input, bytes + guard, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!std::equal(x.begin(), x.end(), y.begin()) || !std::all_of(y.begin() + elements * 2, y.end(), IsGuardWord))
            throw std::runtime_error("input modified");
        std::cout << "PASS bitwise rows=" << h << " cols=" << w << " batch=" << b << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TRANSPOSE_FAILURE " << error.what() << '\n';
        return 1;
    }
}
