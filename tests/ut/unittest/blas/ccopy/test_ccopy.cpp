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
#include <complex>
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

std::string GetCcopyOutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    if (current_dir_env) {
        return std::string(current_dir_env);
    } else {
        return std::string("get env failed.");
    }
}
} // namespace

TEST(TestBlasCcopy, TestCcopyCase0)
{
    std::string originPath = GetCcopyOutputDirectory() + "/tests/ut/unittest/blas/ccopy/ccopy_data";
    std::string destPath = GetCcopyOutputDirectory() + "/build/tests/ut/unittest/blas/ccopy";
    std::string mkdir_cmd = "mkdir -p " + ShellQuote(destPath);
    std::string copy_cmd = "cp -rf " + ShellQuote(originPath) + " " + ShellQuote(destPath);
    std::string chmod_cmd = "chmod -R 755 " + ShellQuote(destPath + "/ccopy_data/");
    int mkdir_rc = system(mkdir_cmd.c_str());
    ASSERT_EQ(mkdir_rc, 0) << "data prep command failed: " << mkdir_cmd;

    int copy_rc = system(copy_cmd.c_str());
    ASSERT_EQ(copy_rc, 0) << "data prep command failed: " << copy_cmd;

    int chmod_rc = system(chmod_cmd.c_str());
    ASSERT_EQ(chmod_rc, 0) << "data prep command failed: " << chmod_cmd;

    int deviceId = 0;
    int64_t n = 6;
    int64_t incx = 1;
    int64_t incy = 1;

    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) +
                               ".bin";
        if (SaveTensorToBin(context.inTensors[i], destPath + "/ccopy_data/" + filename)) {
            std::cout << "Tensor saved successfully: " << filename << std::endl;
        } else {
            std::cerr << "Failed to save tensor: " << filename << std::endl;
        }
    }

    std::string gen_data_cmd = "cd " + ShellQuote(destPath + "/ccopy_data/") + " && python3 gen_data.py";
    int gen_data_rc = system(gen_data_cmd.c_str());
    ASSERT_EQ(gen_data_rc, 0) << "data prep command failed: " << gen_data_cmd;

    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    std::vector<std::complex<float>> inputHostData(n);
    std::complex<float>* inputDataPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (int64_t i = 0; i < n; i++) {
        inputHostData[i] = inputDataPtr[i];
    }

    std::vector<std::complex<float>> outputHostData(n, std::complex<float>(0.0f, 0.0f));
    std::vector<int64_t> shape = {n};

    auto ret = CreateAclTensor(inputHostData, shape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, shape, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutput);
    ASSERT_EQ(ret, 0);

    asdBlasHandle handle;
    AspbStatus blasStatus = asdBlasCreate(handle);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    blasStatus = asdBlasMakeCopyPlan(handle);
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

    blasStatus = asdBlasCcopy(handle, n, aclInput, incx, aclOutput, incy);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    blasStatus = asdBlasSynchronize(handle);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdBlasDestroy(handle);

    ret = aclrtMemcpy(outputTensor.hostData, outputTensor.dataSize, outputDeviceAddr, outputTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    std::string outFilename = GetElementDtype(outputTensor.desc.dtype) + "_output_.bin";
    if (SaveOutTensorToBin(outputTensor, destPath + "/ccopy_data/" + outFilename)) {
        std::cout << "Output tensor saved successfully" << std::endl;
    } else {
        std::cerr << "Failed to save output tensor" << std::endl;
    }

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    OpTestEnd(deviceId, context, stream);

    std::string cmp_data_cmd = "cd " + ShellQuote(destPath + "/ccopy_data/") + " && python3 compare_data.py";
    int res = system(cmp_data_cmd.c_str());
    std::cout << "compare result = " << res << std::endl;
    ASSERT_EQ(res, 0);
}
