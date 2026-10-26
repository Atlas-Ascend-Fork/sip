/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// SmatinvBatchedOperation C++ 调用示例: 批量实数矩阵求逆 (fp32)
// n < 32 走 Gauss-Jordan 消元 kernel; 32 <= n <= 256 走 LU 分解求逆 kernel
// (内部自动分派)。示例取 n=64 演示 LU 路径。

#include <cstdlib>
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;
    int deviceId = 0;
    const char* envDeviceId = std::getenv("ASDSIP_DEVICE_ID");
    if (envDeviceId != nullptr) {
        deviceId = std::atoi(envDeviceId);
    }

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 演示规模: batch=2, n=64 (LU 分解路径); 可按需修改 (n < 256)
    const int64_t batch = 2;
    const int64_t n = 64;
    const int64_t lda = n;

    // 构造对角占优可逆矩阵: A = randn * 0.1 + diag(n)
    std::vector<float> aHost(batch * n * n, 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                aHost[b * n * n + i * n + j] = 0.1f * ((i * n + j + b) % 7 - 3.0f);
            }
            aHost[b * n * n + i * n + i] += static_cast<float>(n);
        }
    }
    std::vector<float> ainvHost(batch * n * n, 0.0f);

    std::vector<int64_t> matShape = {batch, n, n};
    aclTensor* matA = nullptr;
    aclTensor* matAinv = nullptr;
    aclTensor* info = nullptr; // 算子定义占位, 不生效
    void* matADeviceAddr = nullptr;
    void* matAinvDeviceAddr = nullptr;

    ret = CreateAclTensor(aHost, matShape, &matADeviceAddr, ACL_FLOAT, &matA);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("CreateAclTensor matA failed. ERROR: %d\n", ret); return ret);
    ret = CreateAclTensor(ainvHost, matShape, &matAinvDeviceAddr, ACL_FLOAT, &matAinv);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("CreateAclTensor matAinv failed. ERROR: %d\n", ret); return ret);

    asdBlasHandle handle;
    ASD_STATUS_CHECK(asdBlasCreate(handle));
    ASD_STATUS_CHECK(asdBlasMakeSmatinvBatchedPlan(handle, n, batch));

    // workspace: plan 侧按 n/dtype 估算 (LU 路径含 rsv 区间), 必须在执行前绑定
    size_t lwork = 0;
    void* workspaceBuffer = nullptr;
    ASD_STATUS_CHECK(asdBlasGetWorkspaceSize(handle, lwork));
    if (lwork > 0) {
        ret = aclrtMalloc(&workspaceBuffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
        ASD_STATUS_CHECK(asdBlasSetWorkspace(handle, workspaceBuffer));
    }
    ASD_STATUS_CHECK(asdBlasSetStream(handle, stream));

    // 同步执行 (示例简洁); 生产场景建议异步下发 + 事件管理
    ASD_STATUS_CHECK(asdBlasSmatinvBatched(handle, n, matA, lda, matAinv, lda, info, batch));
    std::cout << "Execute asdBlasSmatinvBatched successfully." << std::endl;

    // 等待算子执行完成后回传校验: || A @ Ainv - I || (应接近 0)
    ASD_STATUS_CHECK(asdBlasSynchronize(handle));
    std::vector<float> ainvOut(batch * n * n, 0.0f);
    ret = aclrtMemcpy(ainvOut.data(), ainvOut.size() * sizeof(float), matAinvDeviceAddr, ainvOut.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);

    double maxResidual = 0.0;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int64_t k = 0; k < n; ++k) {
                    sum += static_cast<double>(aHost[b * n * n + i * n + k]) *
                           static_cast<double>(ainvOut[b * n * n + k * n + j]);
                }
                double expect = (i == j) ? 1.0 : 0.0;
                maxResidual = std::max(maxResidual, std::abs(sum - expect));
            }
        }
    }
    std::cout << "max ||A @ Ainv - I|| = " << maxResidual << " (fp32 量级应 < 1e-4)" << std::endl;

    asdBlasDestroy(handle);
    aclDestroyTensor(matA);
    aclDestroyTensor(matAinv);
    aclrtFree(matADeviceAddr);
    aclrtFree(matAinvDeviceAddr);
    if (workspaceBuffer != nullptr) {
        aclrtFree(workspaceBuffer);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
