/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <complex>
#include <fstream>
#include <random>
#include <string>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/platform/platform_info.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "fft_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;
using namespace Mki;

namespace {

constexpr double C2C2D_RTOL = 1e-3;
constexpr double C2C2D_ATOL = 1e-4;

std::string GetC2C2DOutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    return current_dir_env != nullptr ? std::string(current_dir_env) : std::string(".");
}

void RunFftC2C2DTest(int64_t sizeX, int64_t sizeY, int64_t batch, asdFftDirection direction, const std::string& suffix)
{
    static std::string destPath = PrepareDataDirOnce(GetC2C2DOutputDirectory(), "fft/2d/c2c", "c2c_2d_data");
    std::string dataDir = destPath + "/c2c_2d_data";

    std::string dirStr = (direction == asdFftDirection::ASCEND_FFT_FORWARD) ? "forward" : "inverse";

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, sizeX, sizeY}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, sizeX, sizeY}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    SaveTensorToBin(inputTensor, dataDir + "/complex64_input_0.bin");

    std::string genDataCmd = "cd " + ShellQuote(dataDir) + " && python3 gen_data_2d.py --sizeX " +
                             std::to_string(sizeX) + " --sizeY " + std::to_string(sizeY) + " --batch " +
                             std::to_string(batch) + " --direction " + dirStr + " --suffix " + suffix;
    // golden 生成失败必须失败用例，否则残留旧参数的 golden 会造成"假通过"
    int genRes = system(genDataCmd.c_str());
    ASSERT_EQ(genRes, 0);
    std::string goldenFile = dataDir + "/complex64_golden_c2c_2d" + suffix + ".bin";

    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    std::vector<std::complex<float>> inputHostData(batch * sizeX * sizeY);
    auto* inputDataPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (size_t i = 0; i < inputHostData.size(); ++i) {
        inputHostData[i] = inputDataPtr[i];
    }
    std::vector<std::complex<float>> outputHostData(batch * sizeX * sizeY, std::complex<float>(0.0f, 0.0f));
    std::vector<int64_t> shape = {batch, sizeX, sizeY};

    auto ret = CreateAclTensor(inputHostData, shape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, shape, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutput);
    ASSERT_EQ(ret, 0);

    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftMakePlan2D(handle, sizeX, sizeY, asdFftType::ASCEND_FFT_C2C, direction,
                                 static_cast<int32_t>(batch));
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    fftStatus = asdFftGetWorkspaceSize(handle, workSize);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, ::ACL_SUCCESS);
        fftStatus = asdFftSetWorkspace(handle, (uint8_t*)workspaceAddr);
        ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);
    }

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    fftStatus = asdFftSetStream(handle, aclStream);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftExecC2C(handle, aclInput, aclOutput);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftSynchronize(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdFftDestroy(handle);

    ret = aclrtMemcpy(outputTensor.hostData, outputTensor.dataSize, outputDeviceAddr, outputTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // 比对必须在 OpTestEnd 之前完成——OpTestEnd 释放 tensor host 内存
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    int64_t numel = batch * sizeX * sizeY;
    int res = CompareGoldenWithOutput(goldenFile, outputTensor.hostData, static_cast<size_t>(numel), true, C2C2D_RTOL,
                                      C2C2D_ATOL, &maxAbsErr, &maxRelErr);
    std::cout << "compare result = " << (res == 0 ? 0 : 1) << " (mismatch=" << res << ", max_abs=" << maxAbsErr
              << ", max_rel=" << maxRelErr << ")" << std::endl;
    ASSERT_EQ(res, 0);

    OpTestEnd(deviceId, context, stream);
}
} // namespace

TEST(TestFftC2C2d, TestC2C2DForwardSquare64) { RunFftC2C2DTest(64, 64, 1, asdFftDirection::ASCEND_FFT_FORWARD, "_b1"); }
TEST(TestFftC2C2d, TestC2C2DInverseSquare64)
{
    RunFftC2C2DTest(64, 64, 1, asdFftDirection::ASCEND_FFT_INVERSE, "_inv");
}
TEST(TestFftC2C2d, TestC2C2DForwardRect128x256)
{
    RunFftC2C2DTest(128, 256, 1, asdFftDirection::ASCEND_FFT_FORWARD, "_r");
}
TEST(TestFftC2C2d, TestC2C2DForwardBatch2) { RunFftC2C2DTest(32, 48, 2, asdFftDirection::ASCEND_FFT_FORWARD, "_b2"); }
TEST(TestFftC2C2d, TestC2C2DForwardRadix7x13)
{
    RunFftC2C2DTest(112, 104, 1, asdFftDirection::ASCEND_FFT_FORWARD, "_p");
}

TEST(TestFftC2C2d, TestC2C2DForward500x50)
{
    RunFftC2C2DTest(500, 50, 2, asdFftDirection::ASCEND_FFT_FORWARD, "_f500");
}
TEST(TestFftC2C2d, TestC2C2DInverse500x50)
{
    RunFftC2C2DTest(500, 50, 2, asdFftDirection::ASCEND_FFT_INVERSE, "_i500");
}

// ---- 负向校验（issue #352）：任一维长度为 1 的 2D C2C 不得进入 arch35 混合基核 ----
// 该路径对长度为 1 的维分解出空 radix 表，stage 循环零迭代、输出缓冲不写入而 exec 仍返回
// 成功；修复后在 plan 创建期显式拒绝。950 上断言拒绝；非 950 平台不走该分支，ordinary
// 组合路径的 plan 创建保持可用（回归保护，实测 910B 创建成功）。

void RunMakePlan2DSizeOneRejected(int64_t sizeX, int64_t sizeY)
{
    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(
        asdFftMakePlan2D(handle, sizeX, sizeY, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
        AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    asdFftDestroy(handle);
}

TEST(TestFftC2C2d, TestC2C2DSizeOneDimArch35Admission)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        RunMakePlan2DSizeOneRejected(1, 500);
        RunMakePlan2DSizeOneRejected(500, 1);
        RunMakePlan2DSizeOneRejected(1, 12);
    } else {
        asdFftHandle handle;
        ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
        EXPECT_EQ(asdFftMakePlan2D(handle, 1, 500, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
                  AsdSip::ErrorType::ACL_SUCCESS);
        asdFftDestroy(handle);
    }

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}
