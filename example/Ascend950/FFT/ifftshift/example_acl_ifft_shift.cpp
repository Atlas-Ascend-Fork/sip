/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            return -1;                                   \
        }                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    // 优先从环境变量ASDSIP_DEVICE_ID 获取实际 device id(build.sh 由CI 映射卡注入), 未设置时默认 0
    const char* envDeviceId = std::getenv("ASDSIP_DEVICE_ID");
    if (envDeviceId != nullptr) {
        deviceId = std::atoi(envDeviceId);
    }
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 创造tensor的Host侧数据: batch 个 row x col 复数切片
    int64_t batch = 2, row = 4, col = 6;
    int64_t dist = 0; // 0 表示输入切片连续 (dist = row * col)
    // scale = 1/(row*col): 演示将逆 FFT 归一化折叠进移位 pass; scale = 1.0 时为纯逆排列
    const double scale = 1.0 / static_cast<double>(row * col);
    const float scale32 = static_cast<float>(scale);
    const int64_t tensorSize = batch * row * col;
    std::vector<int64_t> inShape = {batch, row, col};
    std::vector<std::complex<float>> inputHostData(tensorSize, std::complex<float>(0, 0));
    for (int64_t i = 0; i < tensorSize; i++) {
        inputHostData[i] = std::complex<float>(static_cast<float>(i), static_cast<float>(i) + 1.0f);
    }
    std::vector<std::complex<float>> outHostData(tensorSize, std::complex<float>(0, 0));
    void* inputDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &input);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, inShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    // 执行 IfftShift: output[s][(r + row/2) % row][(c + col/2) % col] = scale * input[s][r][c]
    ASD_STATUS_CHECK(asdIfftShift(row, col, batch, dist, scale, input, out, stream));
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<std::complex<float>> outData(tensorSize, std::complex<float>(0, 0));
    ret = aclrtMemcpy(outData.data(), tensorSize * sizeof(outData[0]), outDeviceAddr, tensorSize * sizeof(outData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Host侧golden: 逆排列 + float32 逐实部/虚部缩放, 逐元素二进制比对
    const int64_t shiftR = row / 2;
    const int64_t shiftC = col / 2;
    int64_t diff = 0;
    for (int64_t s = 0; s < batch; s++) {
        for (int64_t r = 0; r < row; r++) {
            for (int64_t c = 0; c < col; c++) {
                auto& srcv = inputHostData[s * row * col + r * col + c];
                auto expect = std::complex<float>(srcv.real() * scale32, srcv.imag() * scale32);
                auto got = outData[s * row * col + (r + shiftR) % row * col + (c + shiftC) % col];
                if (expect != got) {
                    diff++;
                }
            }
        }
    }
    std::cout << "IfftShift result: " << (diff == 0 ? "PASS" : "FAIL") << " (";
    std::cout << tensorSize - diff << "/" << tensorSize << " elements exact)" << std::endl;
    // 打印输出tensor值中前8个
    for (int64_t i = 0; i < 8; i++) {
        std::cout << outData[i] << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(input);
    aclDestroyTensor(out);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
