/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            return -1;                                   \
        }                                                \
    } while (0)

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
    // 固定写法，acl初始化
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

void PrintMatrix(const std::vector<float>& data, int64_t batch, int64_t rows, int64_t cols)
{
    for (int64_t b = 0; b < batch; b++) {
        std::cout << "batch " << b << ":" << std::endl;
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                std::cout << data[(b * rows + i) * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

int main(int argc, char** argv)
{
    int deviceId = 0;
    // 优先从环境变量 ASDSIP_DEVICE_ID 获取实际 device id(build.sh 按 CI 映射卡注入), 未设置时默认 0
    const char* envDeviceId = std::getenv("ASDSIP_DEVICE_ID");
    if (envDeviceId != nullptr) {
        deviceId = std::atoi(envDeviceId);
    }

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 批量求解 A * X = B, batch=2, n=3, nrhs=2
    int64_t batch = 2;
    int64_t n = 3;
    int64_t nrhs = 2;

    // A = [[4,1,0],[1,3,1],[0,1,2]]（两个批次相同）
    std::vector<float> tensorInAData = {4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f,
                                        4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f};
    // B 由已知解反算: X0 = [[1,2],[3,4],[5,6]], X1 = [[2,1],[1,2],[3,4]]
    std::vector<float> tensorInBData = {7.0f, 12.0f, 15.0f, 20.0f, 13.0f, 16.0f, 9.0f, 6.0f, 8.0f, 11.0f, 7.0f, 10.0f};
    std::vector<float> tensorOutXData(batch * n * nrhs, 0.0f);

    std::cout << "------- input A -------" << std::endl;
    PrintMatrix(tensorInAData, batch, n, n);
    std::cout << "------- input B -------" << std::endl;
    PrintMatrix(tensorInBData, batch, n, nrhs);

    std::vector<int64_t> aShape = {batch, n, n};
    std::vector<int64_t> bShape = {batch, n, nrhs};
    std::vector<int64_t> xShape = {batch, n, nrhs};

    aclTensor* inputA = nullptr;
    aclTensor* inputB = nullptr;
    aclTensor* outputX = nullptr;
    void* inputADeviceAddr = nullptr;
    void* inputBDeviceAddr = nullptr;
    void* outputXDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInBData, bShape, &inputBDeviceAddr, aclDataType::ACL_FLOAT, &inputB);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutXData, xShape, &outputXDeviceAddr, aclDataType::ACL_FLOAT, &outputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    size_t lwork = 0;
    ASD_STATUS_CHECK(solveGetWorkspaceSize(lwork));
    std::cout << "lwork = " << lwork << std::endl;
    ASD_STATUS_CHECK(solve(inputA, inputB, outputX, stream, nullptr));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutXData.data(), GetShapeSize(xShape) * sizeof(float), outputXDeviceAddr,
                      GetShapeSize(xShape) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy x from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output X -------" << std::endl;
    PrintMatrix(tensorOutXData, batch, n, nrhs);

    // 残差校验: max|A*X - B|
    float maxResidual = 0.0f;
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < n; i++) {
            for (int64_t c = 0; c < nrhs; c++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < n; k++) {
                    sum += tensorInAData[(b * n + i) * n + k] * tensorOutXData[(b * n + k) * nrhs + c];
                }
                maxResidual = std::max(maxResidual, std::fabs(sum - tensorInBData[(b * n + i) * nrhs + c]));
            }
        }
    }
    std::cout << "max|A*X - B| = " << maxResidual << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputA);
    aclDestroyTensor(inputB);
    aclDestroyTensor(outputX);
    aclrtFree(inputADeviceAddr);
    aclrtFree(inputBDeviceAddr);
    aclrtFree(outputXDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
