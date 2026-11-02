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
#include <filesystem>
#include <cmath>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "blas_api.h"
#include "utils/mem_base_inner.h"
using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;

std::string GetSdotOutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    if (current_dir_env) {
        return std::string(current_dir_env);
    } else {
        return std::string("get env failed.");
    }
}
} // namespace

TEST(TestBlasSdot, TestSdotCase0)
{
    std::string originPath = GetSdotOutputDirectory() + "/tests/ut/unittest/blas/sdot/sdot_data";
    std::string destPath = GetSdotOutputDirectory() + "/build/tests/ut/unittest/blas/sdot";
    std::string mkdir_cmd = "mkdir -p " + ShellQuote(destPath);
    std::string copy_cmd = "cp -rf " + ShellQuote(originPath) + " " + ShellQuote(destPath);
    std::string chmod_cmd = "chmod -R 755 " + ShellQuote(destPath + "/sdot_data/");
    system(mkdir_cmd.c_str());
    system(copy_cmd.c_str());
    system(chmod_cmd.c_str());

    int deviceId = 0;
    int64_t n = 7;
    int64_t incx = 1;
    int64_t incy = 1;

    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensorX = context.inTensors[0];
    Tensor inputTensorY = context.inTensors[1];
    Tensor resultTensor = context.inTensors[2];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) +
                               ".bin";
        if (SaveTensorToBin(context.inTensors[i], destPath + "/sdot_data/" + filename)) {
            std::cout << "Tensor saved successfully: " << filename << std::endl;
        } else {
            std::cerr << "Failed to save tensor: " << filename << std::endl;
        }
    }

    std::string gen_data_cmd = "cd " + ShellQuote(destPath + "/sdot_data/") + " && python3 gen_data.py";
    system(gen_data_cmd.c_str());

    aclTensor* aclInputX = nullptr;
    aclTensor* aclInputY = nullptr;
    aclTensor* aclResult = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;

    std::vector<float> inputXHostData(n);
    float* inputXDataPtr = static_cast<float*>(inputTensorX.hostData);
    for (int64_t i = 0; i < n; i++) {
        inputXHostData[i] = inputXDataPtr[i];
    }

    std::vector<float> inputYHostData(n);
    float* inputYDataPtr = static_cast<float*>(inputTensorY.hostData);
    for (int64_t i = 0; i < n; i++) {
        inputYHostData[i] = inputYDataPtr[i];
    }

    std::vector<float> resultHostData(1, 0.0f);
    std::vector<int64_t> xShape = {n};
    std::vector<int64_t> yShape = {n};
    std::vector<int64_t> resultShape = {1};

    auto ret = CreateAclTensor(inputXHostData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &aclInputX);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(inputYHostData, yShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &aclInputY);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(resultHostData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &aclResult);
    ASSERT_EQ(ret, 0);

    asdBlasHandle handle;
    AspbStatus blasStatus = asdBlasCreate(handle);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    blasStatus = asdBlasMakeDotPlan(handle);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    blasStatus = asdBlasGetWorkspaceSize(handle, workSize);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, ::ACL_SUCCESS);
        blasStatus = asdBlasSetWorkspace(handle, workspaceAddr);
        ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);
    }

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    blasStatus = asdBlasSetStream(handle, aclStream);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    blasStatus = asdBlasSdot(handle, n, aclInputX, incx, aclInputY, incy, aclResult);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    blasStatus = asdBlasSynchronize(handle);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdBlasDestroy(handle);

    ret = aclrtMemcpy(resultTensor.hostData, resultTensor.dataSize, resultDeviceAddr, resultTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    std::string outFilename = GetElementDtype(resultTensor.desc.dtype) + "_output_.bin";
    if (SaveOutTensorToBin(resultTensor, destPath + "/sdot_data/" + outFilename)) {
        std::cout << "Output tensor saved successfully" << std::endl;
    } else {
        std::cerr << "Failed to save output tensor" << std::endl;
    }

    aclDestroyTensor(aclInputX);
    aclDestroyTensor(aclInputY);
    aclDestroyTensor(aclResult);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // dot 族 workspace 跨核归约回读对显存页隔离敏感, 保持模式下回退基线复位
    OpTestMarkResetOnEnd();
    OpTestEnd(deviceId, context, stream);

    std::string cmp_data_cmd = "cd " + ShellQuote(destPath + "/sdot_data/") + " && python3 compare_data.py";
    int res = system(cmp_data_cmd.c_str());
    std::cout << "compare result = " << res << std::endl;
    ASSERT_EQ(res, 0);
}
