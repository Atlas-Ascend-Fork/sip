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
#include "base_api.h"
#include "base_inner_api.h"
#include "blas_api.h"
#include "utils/mem_base_inner.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;

std::string GetAsdMulOutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    if (current_dir_env) {
        return std::string(current_dir_env);
    } else {
        return std::string("get env failed.");
    }
}
} // namespace

TEST(TestOpAsdMul, TestAsdMulComplex64Case0)
{
    // 数据准备统一走 PrepareDataDirOnce：mkdir/cp/chmod 一次执行 + dest 按 shard 隔离，
    // 消除双 shard 并发对同一 dest 的 cp 竞争假失败（issue #417）
    static std::string destPath = PrepareDataDirOnce(GetAsdMulOutputDirectory(), "base/asd_mul", "asd_mul_data");
    ASSERT_TRUE(std::filesystem::exists(destPath + "/asd_mul_data")) << "data prep command failed, dest: " << destPath;

    Status status;
    int64_t n = 16;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});

    TensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputX = context.inTensors[0];
    Tensor inputY = context.inTensors[1];
    Tensor outputZ = context.inTensors[2];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) +
                               ".bin";
        SaveTensorToBin(context.inTensors[i], destPath + "/asd_mul_data/" + filename);
    }

    std::string gen_data_cmd = "cd " + ShellQuote(destPath + "/asd_mul_data/") + " && python3 gen_data.py --dtype c64";
    int gen_data_rc = system(gen_data_cmd.c_str());
    ASSERT_EQ(gen_data_rc, 0) << "data prep command failed: " << gen_data_cmd;

    aclTensor* aclInputX = nullptr;
    aclTensor* aclInputY = nullptr;
    aclTensor* aclOutputZ = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* outputZDeviceAddr = nullptr;

    std::vector<std::complex<float>> inputXHostData(n);
    std::complex<float>* inputXDataPtr = static_cast<std::complex<float>*>(inputX.hostData);
    for (int64_t i = 0; i < n; i++) {
        inputXHostData[i] = inputXDataPtr[i];
    }

    std::vector<std::complex<float>> inputYHostData(n);
    std::complex<float>* inputYDataPtr = static_cast<std::complex<float>*>(inputY.hostData);
    for (int64_t i = 0; i < n; i++) {
        inputYHostData[i] = inputYDataPtr[i];
    }

    std::vector<std::complex<float>> outputZHostData(n, std::complex<float>(0.0f, 0.0f));
    std::vector<int64_t> shape = {n};

    auto ret = CreateAclTensor(inputXHostData, shape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInputX);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(inputYHostData, shape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInputY);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputZHostData, shape, &outputZDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutputZ);
    ASSERT_EQ(ret, 0);

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    AspbStatus blasStatus = asdMul(n, aclInputX, aclInputY, aclOutputZ, aclStream);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    MkiRtStreamSynchronize(stream);

    ret = aclrtMemcpy(outputZ.hostData, outputZ.dataSize, outputZDeviceAddr, outputZ.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    std::string filename = GetElementDtype(outputZ.desc.dtype) + "_output_" + ".bin";
    SaveOutTensorToBin(outputZ, destPath + "/asd_mul_data/" + filename);

    aclDestroyTensor(aclInputX);
    aclDestroyTensor(aclInputY);
    aclDestroyTensor(aclOutputZ);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(outputZDeviceAddr);

    OpTestEnd(deviceId, context, stream);

    std::string cmp_data_cmd = "cd " + ShellQuote(destPath + "/asd_mul_data/") +
                               " && python3 compare_data.py --dtype c64";
    int res = system(cmp_data_cmd.c_str());
    std::cout << "compare result = " << res << std::endl;
    ASSERT_EQ(res, 0);
}

TEST(TestOpAsdMul, TestAsdMulComplex32Case0)
{
    // 数据准备统一走 PrepareDataDirOnce：mkdir/cp/chmod 一次执行 + dest 按 shard 隔离，
    // 消除双 shard 并发对同一 dest 的 cp 竞争假失败（issue #417）
    static std::string destPath = PrepareDataDirOnce(GetAsdMulOutputDirectory(), "base/asd_mul", "asd_mul_data");
    ASSERT_TRUE(std::filesystem::exists(destPath + "/asd_mul_data")) << "data prep command failed, dest: " << destPath;

    Status status;
    int64_t n = 16;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {n}});

    TensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputX = context.inTensors[0];
    Tensor inputY = context.inTensors[1];
    Tensor outputZ = context.inTensors[2];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) +
                               ".bin";
        SaveTensorToBin(context.inTensors[i], destPath + "/asd_mul_data/" + filename);
    }

    std::string gen_data_cmd = "cd " + ShellQuote(destPath + "/asd_mul_data/") + " && python3 gen_data.py --dtype c32";
    int gen_data_rc = system(gen_data_cmd.c_str());
    ASSERT_EQ(gen_data_rc, 0) << "data prep command failed: " << gen_data_cmd;

    aclTensor* aclInputX = nullptr;
    aclTensor* aclInputY = nullptr;
    aclTensor* aclOutputZ = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* outputZDeviceAddr = nullptr;

    std::vector<int64_t> shape = {n};
    std::vector<int64_t> strides = {1};

    auto ret = aclrtMalloc(&inputXDeviceAddr, inputX.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    ret = aclrtMemcpy(inputXDeviceAddr, inputX.dataSize, inputX.hostData, inputX.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    ret = aclrtMalloc(&inputYDeviceAddr, inputY.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    ret = aclrtMemcpy(inputYDeviceAddr, inputY.dataSize, inputY.hostData, inputY.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    ret = aclrtMalloc(&outputZDeviceAddr, outputZ.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    aclInputX = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_COMPLEX32, strides.data(), 0,
                                aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), inputXDeviceAddr);
    ASSERT_NE(aclInputX, nullptr);
    aclInputY = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_COMPLEX32, strides.data(), 0,
                                aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), inputYDeviceAddr);
    ASSERT_NE(aclInputY, nullptr);
    aclOutputZ = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_COMPLEX32, strides.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), outputZDeviceAddr);
    ASSERT_NE(aclOutputZ, nullptr);

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    AspbStatus blasStatus = asdMul(n, aclInputX, aclInputY, aclOutputZ, aclStream);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    MkiRtStreamSynchronize(stream);

    ret = aclrtMemcpy(outputZ.hostData, outputZ.dataSize, outputZDeviceAddr, outputZ.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    std::string filename = GetElementDtype(outputZ.desc.dtype) + "_output_" + ".bin";
    SaveOutTensorToBin(outputZ, destPath + "/asd_mul_data/" + filename);

    aclDestroyTensor(aclInputX);
    aclDestroyTensor(aclInputY);
    aclDestroyTensor(aclOutputZ);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(outputZDeviceAddr);

    OpTestEnd(deviceId, context, stream);

    std::string cmp_data_cmd = "cd " + ShellQuote(destPath + "/asd_mul_data/") +
                               " && python3 compare_data.py --dtype c32";
    int res = system(cmp_data_cmd.c_str());
    std::cout << "compare result = " << res << std::endl;
    ASSERT_EQ(res, 0);
}
