/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdlib>
#include <vector>
#include <mki/utils/rt/rt.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/ops_base.h"
#include "params/fft_c2c_arch35.h"
#include "../../../ops/fft/common/include/tiling/fft_all_mix_tiling_data.h"
#include "fftcore/fft_c2c2d_arch35_core.h"
#include "utils/fft_gemm_coeffs.h"

constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = K_PI * 2;

using namespace AsdSip;

namespace {
constexpr int64_t MIXED_RADIX_THRESHOLD = 1 << 14;
constexpr int64_t MULTI_CORE_BATCH_THRESHOLD = 32;

int32_t Log2PowerOfTwo(int64_t value)
{
    int32_t log = 0;
    while (value > 1) {
        value >>= 1;
        ++log;
    }
    return log;
}

bool IsPowerOfTwo(int64_t value) { return value > 1 && (value & (value - 1)) == 0; }

bool ShouldUseMixedRadixKernel(int64_t fftLen, int64_t batchSize)
{
    return !IsPowerOfTwo(fftLen) || fftLen < MIXED_RADIX_THRESHOLD || batchSize <= MULTI_CORE_BATCH_THRESHOLD;
}

void FreeDeviceAddrs(std::vector<uint8_t*>& deviceAddrs)
{
    for (uint8_t* deviceAddr : deviceAddrs) {
        if (deviceAddr != nullptr) {
            MkiRtMemFreeDevice(deviceAddr);
        }
    }
    deviceAddrs.clear();
}

size_t GetRadix2TwiddleFloatCount(int64_t n)
{
    size_t totalTwComplex = 0;
    for (int64_t len = 2; len <= n; len <<= 1) {
        totalTwComplex += static_cast<size_t>(len >> 1);
    }
    return totalTwComplex * 2;
}

AsdSip::FFTensor* BuildRadix2TwiddleTensor(int64_t n, bool forward)
{
    double sign = forward ? -1.0 : 1.0;
    size_t totalTwFloats = GetRadix2TwiddleFloatCount(n);
    float* host = new float[totalTwFloats]();

    size_t offset = 0;
    for (int64_t len = 2; len <= n; len <<= 1) {
        int64_t half = len >> 1;
        for (int64_t j = 0; j < half; j++) {
            double angle = sign * K_2PI * j / len;
            host[(offset + j) * 2] = static_cast<float>(std::cos(angle));
            host[(offset + j) * 2 + 1] = static_cast<float>(std::sin(angle));
        }
        offset += static_cast<size_t>(half);
    }

    AsdSip::FFTensor* tensor = new AsdSip::FFTensor;
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT,
                    Mki::TensorFormat::TENSOR_FORMAT_ND,
                    {static_cast<int64_t>(totalTwFloats)},
                    {},
                    0};
    tensor->hostData = host;
    tensor->dataSize = totalTwFloats * sizeof(float);
    return tensor;
}

std::vector<int32_t> FactorizeMixed(int64_t n)
{
    std::vector<int32_t> radices;
    constexpr int32_t allowedRadices[] = {2, 3, 5, 7, 11, 13, 17, 19};
    for (int32_t radix : allowedRadices) {
        while (n % radix == 0) {
            radices.push_back(radix);
            n /= radix;
        }
    }
    if (n != 1) {
        ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 unsupported mixed-radix factor remains: " << n;
        throw std::runtime_error("FftC2C2DCoreArch35 unsupported mixed-radix factorization.");
    }
    return radices;
}

AsdSip::FFTensor* BuildRadixListTensor(const std::vector<int32_t>& radices)
{
    AsdSip::FFTensor* tensor = new AsdSip::FFTensor;
    int32_t* host = new int32_t[radices.size()];
    for (size_t i = 0; i < radices.size(); ++i) {
        host[i] = radices[i];
    }
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_INT32,
                    Mki::TensorFormat::TENSOR_FORMAT_ND,
                    {static_cast<int64_t>(radices.size())},
                    {},
                    0};
    tensor->hostData = host;
    tensor->dataSize = radices.size() * sizeof(int32_t);
    return tensor;
}

AsdSip::FFTensor* BuildRadix2DftTensor()
{
    AsdSip::FFTensor* tensor = new AsdSip::FFTensor;
    float* host = new float[1]();
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {1}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = sizeof(float);
    return tensor;
}

AsdSip::FFTensor* BuildMixedTwiddleTensor(const std::vector<int32_t>& radices, bool forward)
{
    double sign = forward ? -1.0 : 1.0;
    int64_t totalTwComplex = 0;
    int64_t prev = 1;
    for (int32_t radix : radices) {
        totalTwComplex += radix * prev;
        prev *= radix;
    }

    float* host = new float[static_cast<size_t>(totalTwComplex) * 2]();
    prev = 1;
    int64_t len = 1;
    int64_t off = 0;
    for (int32_t radix : radices) {
        len *= radix;
        for (int32_t p = 0; p < radix; ++p) {
            for (int64_t j = 0; j < prev; ++j) {
                double angle = sign * K_2PI * p * j / len;
                host[(off + p * prev + j) * 2] = static_cast<float>(std::cos(angle));
                host[(off + p * prev + j) * 2 + 1] = static_cast<float>(std::sin(angle));
            }
        }
        off += radix * prev;
        prev = len;
    }

    AsdSip::FFTensor* tensor = new AsdSip::FFTensor;
    tensor->desc = {
        Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {totalTwComplex * 2}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = static_cast<size_t>(totalTwComplex) * 2 * sizeof(float);
    return tensor;
}

AsdSip::FFTensor* BuildMixedDftTensor(const std::vector<int32_t>& radices, bool forward)
{
    double sign = forward ? -1.0 : 1.0;
    int64_t totalDftComplex = 0;
    for (int32_t radix : radices) {
        totalDftComplex += radix * radix;
    }

    float* host = new float[static_cast<size_t>(totalDftComplex) * 2]();
    int64_t off = 0;
    for (int32_t radix : radices) {
        for (int32_t q = 0; q < radix; ++q) {
            for (int32_t p = 0; p < radix; ++p) {
                double angle = sign * K_2PI * p * q / radix;
                host[(off + q * radix + p) * 2] = static_cast<float>(std::cos(angle));
                host[(off + q * radix + p) * 2 + 1] = static_cast<float>(std::sin(angle));
            }
        }
        off += radix * radix;
    }

    AsdSip::FFTensor* tensor = new AsdSip::FFTensor;
    tensor->desc = {
        Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {totalDftComplex * 2}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = static_cast<size_t>(totalDftComplex) * 2 * sizeof(float);
    return tensor;
}

void FillRadix2StageTiling(FftC2CArch35StageTilingData& stageTiling, int64_t stageBatch, int64_t fftLen, int32_t stage,
                           int32_t outer, int32_t isInverse, bool isLast)
{
    int64_t nHalf = fftLen >> 1;
    int32_t len = 1 << (stage + 1);
    int32_t half = len >> 1;

    stageTiling = {};
    stageTiling.batch = stageBatch;
    stageTiling.n = fftLen;
    stageTiling.totalButterflies = stageBatch * nHalf;
    stageTiling.nHalf = static_cast<int32_t>(nHalf);
    stageTiling.len = len;
    stageTiling.half = half;
    stageTiling.logNhalf = Log2PowerOfTwo(nHalf);
    stageTiling.logHalf = stage;
    stageTiling.twOffset = half - 1;
    stageTiling.outer = outer;
    stageTiling.isInverse = isInverse;
    stageTiling.scaleOut = 0;
    stageTiling.transpose = isLast ? 1 : 0;
}

void FillMixed1DPassTiling(FftAllMixTilingData& tiling, int64_t stageBatch, int64_t fftLen, int32_t radixLen,
                           int32_t isInverse, int32_t outer, int32_t transpose)
{
    tiling = {};
    tiling.batchSize = stageBatch;
    tiling.fftN = fftLen;
    tiling.radixListLen = radixLen;
    tiling.isInverse = isInverse;
    tiling.outer = outer;
    tiling.scaleOut = 0;
    tiling.transpose = transpose;

    int64_t singleBufferSize = stageBatch * fftLen * 2 * static_cast<int64_t>(sizeof(float));
    tiling.workspaceOffsets[0] = 0;
    tiling.workspaceOffsets[1] = singleBufferSize;
}

} // namespace

size_t FftC2C2DCoreArch35::EstimateWorkspaceSize()
{
    int64_t singleBufferSize = static_cast<int64_t>(problemDesc.batch) * fftN * fftM * 2 * sizeof(float);
    bool hasMixedPass = ShouldUseMixedRadixKernel(fftM, static_cast<int64_t>(problemDesc.batch) * fftN) ||
                        ShouldUseMixedRadixKernel(fftN, static_cast<int64_t>(problemDesc.batch) * fftM);
    int64_t workspaceSize = getAlignedSize((hasMixedPass ? 3 : 2) *
                                           singleBufferSize); // allocate 内部 32B 上取整,池须同口径
    ASDSIP_LOG(INFO) << "ASCEND_950 FftC2C2DCoreArch35 workspace size: " << workspaceSize;
    return static_cast<size_t>(workspaceSize);
}

void FftC2C2DCoreArch35::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    int64_t batch = static_cast<int64_t>(problemDesc.batch);
    int64_t fullComplexFloats = batch * fftN * fftM * 2;
    bool hasMixedPass = useMixedM || useMixedN;
    int64_t workspaceBuffers = hasMixedPass ? 3 : 2;
    int64_t workspaceBytes = workspaceBuffers * fullComplexFloats * static_cast<int64_t>(sizeof(float));
    float* intermediate = static_cast<float*>(workspace.allocate(static_cast<size_t>(workspaceBytes)));
    float* scratch = intermediate + fullComplexFloats;

    runInfo.SetScratchDeviceAddr(reinterpret_cast<uint8_t*>(hasMixedPass ? scratch : intermediate));
    runInfo.SetStream(stream);

    int32_t isInverse = 1 - static_cast<int32_t>(problemDesc.forward);
    auto launchMixedPass = [&](std::unique_ptr<Mki::Kernel>& passKernel, const std::vector<uint8_t*>& tilingAddrs,
                               void* stageSrc, void* stageDst, const std::shared_ptr<AsdSip::FFTensor>& twTensor,
                               const std::shared_ptr<AsdSip::FFTensor>& radixTensor,
                               const std::shared_ptr<AsdSip::FFTensor>& dftTensor,
                               const FftAllMixTilingData& passTiling) -> bool {
        if (passKernel == nullptr || tilingAddrs.empty()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 mixed-radix pass kernel or tiling buffer is not initialized.";
            return false;
        }

        uint8_t* tilingDeviceAddr = tilingAddrs[0];
        int st = MkiRtMemCopy(tilingDeviceAddr, sizeof(passTiling), &passTiling, sizeof(passTiling),
                              MKIRT_MEMCOPY_HOST_TO_DEVICE);
        if (st != MKIRT_SUCCESS) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 copy mixed pass tiling failed.";
            return false;
        }

        OpParam::FftC2CArch35 passParam = {passTiling.fftN, passTiling.batchSize, passTiling.radixListLen,
                                           passTiling.isInverse, true};

        Tensor passIn;
        Tensor passOut;
        passIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {passTiling.batchSize, passTiling.fftN}, {}, 0};
        passIn.dataSize = static_cast<size_t>(passTiling.batchSize) * static_cast<size_t>(passTiling.fftN) *
                          GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        passIn.data = stageSrc;
        passOut.desc = passIn.desc;
        passOut.dataSize = passIn.dataSize;
        passOut.data = stageDst;

        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(passParam);
        passLaunchParam.AddInTensor(passIn);
        passLaunchParam.AddInTensor(*dftTensor);
        passLaunchParam.AddInTensor(*twTensor);
        passLaunchParam.AddInTensor(*radixTensor);
        passLaunchParam.AddOutTensor(passOut);
        runInfo.SetTilingDeviceAddr(tilingDeviceAddr);

        auto status = passKernel->Run(passLaunchParam, runInfo);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 mixed pass launch failed.";
            return false;
        }
        return true;
    };

    auto launchRadix2Stage = [&](std::unique_ptr<Mki::Kernel>& passKernel, uint8_t* stageTilingDeviceAddr,
                                 void* stageSrc, void* stageDst, const std::shared_ptr<AsdSip::FFTensor>& twTensor,
                                 const std::shared_ptr<AsdSip::FFTensor>& radixTensor,
                                 const std::shared_ptr<AsdSip::FFTensor>& dftTensor, int64_t radixListLen,
                                 const FftC2CArch35StageTilingData& stageTiling) -> bool {
        if (passKernel == nullptr || stageTilingDeviceAddr == nullptr) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 radix-2 pass kernel or tiling buffer is not initialized.";
            return false;
        }

        int st = MkiRtMemCopy(stageTilingDeviceAddr, sizeof(stageTiling), &stageTiling, sizeof(stageTiling),
                              MKIRT_MEMCOPY_HOST_TO_DEVICE);
        if (st != MKIRT_SUCCESS) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 copy stage tiling failed.";
            return false;
        }

        OpParam::FftC2CArch35 passParam = {stageTiling.n, stageTiling.batch, radixListLen, stageTiling.isInverse,
                                           false};

        Tensor passIn;
        Tensor passOut;
        passIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {stageTiling.batch, stageTiling.n}, {}, 0};
        passIn.dataSize = static_cast<size_t>(stageTiling.batch) * static_cast<size_t>(stageTiling.n) *
                          GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        passIn.data = stageSrc;
        passOut.desc = passIn.desc;
        passOut.dataSize = passIn.dataSize;
        passOut.data = stageDst;

        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(passParam);
        passLaunchParam.AddInTensor(passIn);
        passLaunchParam.AddInTensor(*dftTensor);
        passLaunchParam.AddInTensor(*twTensor);
        passLaunchParam.AddInTensor(*radixTensor);
        passLaunchParam.AddOutTensor(passOut);
        runInfo.SetTilingDeviceAddr(stageTilingDeviceAddr);

        auto status = passKernel->Run(passLaunchParam, runInfo);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 radix-2 stage launch failed.";
            return false;
        }
        return true;
    };

    auto launchHybridPass = [&](std::unique_ptr<Mki::Kernel>& passKernel, uint8_t* tilingAddr, void* stageSrc,
                                void* stageDst, const std::shared_ptr<AsdSip::FFTensor>& wTensor) -> bool {
        if (passKernel == nullptr || tilingAddr == nullptr) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 hybrid pass kernel or tiling buffer is not initialized.";
            return false;
        }

        // 参数打包(脚手架约定):fftN=2D fftM(短维 L),batchSize=rows,radixListLen=2D fftN(瓦片行数)
        OpParam::FftC2CArch35 passParam = {fftM, batch * fftN, fftN, isInverse, false, false, false, true};
        Tensor passIn;
        Tensor passOut;
        passIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch * fftN, fftM}, {}, 0};
        passIn.dataSize = static_cast<size_t>(batch * fftN) * static_cast<size_t>(fftM) *
                          GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        passIn.data = stageSrc;
        passOut.desc = passIn.desc;
        passOut.dataSize = passIn.dataSize;
        passOut.data = stageDst;

        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(passParam);
        passLaunchParam.AddInTensor(passIn);
        passLaunchParam.AddInTensor(*wTensor);
        passLaunchParam.AddOutTensor(passOut);
        runInfo.SetTilingDeviceAddr(tilingAddr);

        auto status = passKernel->Run(passLaunchParam, runInfo);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 hybrid pass launch failed.";
            return false;
        }
        return true;
    };

    auto launchGemmPass = [&](std::unique_ptr<Mki::Kernel>& passKernel, uint8_t* tilingAddr, void* stageSrc,
                              void* stageDst, const std::shared_ptr<AsdSip::FFTensor>& wTensor) -> bool {
        if (passKernel == nullptr || tilingAddr == nullptr) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 gemm pass kernel or tiling buffer is not initialized.";
            return false;
        }

        OpParam::FftC2CArch35 passParam = {fftM, batch * fftN, 0, isInverse, false, true};
        Tensor passIn;
        Tensor passOut;
        passIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch * fftN, fftM}, {}, 0};
        passIn.dataSize = static_cast<size_t>(batch * fftN) * static_cast<size_t>(fftM) *
                          GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        passIn.data = stageSrc;
        passOut.desc = passIn.desc;
        passOut.dataSize = passIn.dataSize;
        passOut.data = stageDst;

        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(passParam);
        passLaunchParam.AddInTensor(passIn);
        passLaunchParam.AddInTensor(*wTensor);
        passLaunchParam.AddOutTensor(passOut);
        runInfo.SetTilingDeviceAddr(tilingAddr);

        auto status = passKernel->Run(passLaunchParam, runInfo);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 gemm pass launch failed.";
            return false;
        }
        return true;
    };

    auto launchTransPass = [&](std::unique_ptr<Mki::Kernel>& passKernel, uint8_t* tilingAddr, void* stageSrc,
                               void* stageDst) -> bool {
        if (passKernel == nullptr || tilingAddr == nullptr) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 trans pass kernel or tiling buffer is not initialized.";
            return false;
        }

        OpParam::FftC2CArch35 passParam = {fftN, batch, fftM, isInverse, false, true};
        Tensor passIn;
        Tensor passOut;
        passIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch * fftN, fftM}, {}, 0};
        passIn.dataSize = static_cast<size_t>(batch * fftN) * static_cast<size_t>(fftM) *
                          GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        passIn.data = stageSrc;
        passOut.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch * fftM, fftN}, {}, 0};
        passOut.dataSize = passIn.dataSize;
        passOut.data = stageDst;

        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(passParam);
        passLaunchParam.AddInTensor(passIn);
        passLaunchParam.AddOutTensor(passOut);
        runInfo.SetTilingDeviceAddr(tilingAddr);

        auto status = passKernel->Run(passLaunchParam, runInfo);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 trans pass launch failed.";
            return false;
        }
        return true;
    };

    auto launchNFusedPass = [&](std::unique_ptr<Mki::Kernel>& passKernel, uint8_t* tilingAddr, void* stageSrc,
                                void* stageDst, const std::shared_ptr<AsdSip::FFTensor>& twTensor) -> bool {
        if (passKernel == nullptr || tilingAddr == nullptr) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 nfused pass kernel or tiling buffer is not initialized.";
            return false;
        }

        OpParam::FftC2CArch35 passParam = {fftN, batch, fftM, isInverse, false, false, true};
        Tensor passIn;
        Tensor passOut;
        passIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch * fftM, fftN}, {}, 0};
        passIn.dataSize = static_cast<size_t>(batch * fftM) * static_cast<size_t>(fftN) *
                          GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        passIn.data = stageSrc;
        passOut.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch * fftN, fftM}, {}, 0};
        passOut.dataSize = passIn.dataSize;
        passOut.data = stageDst;

        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(passParam);
        passLaunchParam.AddInTensor(passIn);
        passLaunchParam.AddInTensor(*twTensor);
        passLaunchParam.AddOutTensor(passOut);
        runInfo.SetTilingDeviceAddr(tilingAddr);

        auto status = passKernel->Run(passLaunchParam, runInfo);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 nfused pass launch failed.";
            return false;
        }
        return true;
    };

    if (useHybrid) {
        // 混合核(终态):AIC GEMM→UB 瓦片 + AIV 融合 N 变换直写 GM——一个 mix 核做完整个 2d
        // 无 TRANS、无 N pass、中间结果零 GM 往返
        if (!launchHybridPass(kernelHybrid, tilingHybridDeviceAddrs[0], input, output, hybridCoeff)) {
            workspace.recycleLast();
            return;
        }
        // 计划级首射同步+重发射:mix 核同含 cube 通道,沿 GEMM 惯例缓解 271 首射
        if (!hybridWarmupDone) {
            hybridWarmupDone = true;
            if (MkiRtStreamSynchronize(stream) != MKIRT_SUCCESS) {
                if (!launchHybridPass(kernelHybrid, tilingHybridDeviceAddrs[0], input, output, hybridCoeff)) {
                    workspace.recycleLast();
                    return;
                }
            }
        }
    } else if (useGemmM) {
        // GEMM1: input → scratch(fftM/连续维变换,A 行连续)
        if (!launchGemmPass(kernelGemm, tilingGemmDeviceAddrs[0], input, scratch, dftMatrixM)) {
            workspace.recycleLast();
            return;
        }
        // 计划级首射同步+重发射:cube 通道首射偶发 271 越界,同流重试即恢复
        if (!gemmWarmupDone) {
            gemmWarmupDone = true;
            if (MkiRtStreamSynchronize(stream) != MKIRT_SUCCESS) {
                if (!launchGemmPass(kernelGemm, tilingGemmDeviceAddrs[0], input, scratch, dftMatrixM)) {
                    workspace.recycleLast();
                    return;
                }
            }
        }
        // 转置: scratch [b][x][ỹ] → intermediate [b][ỹ][x](N pass 的输入布局)
        if (!launchTransPass(kernelTrans, tilingTransDeviceAddrs[0], scratch, intermediate)) {
            workspace.recycleLast();
            return;
        }
    } else if (useMixedM) {
        FftAllMixTilingData passM = {};
        FillMixed1DPassTiling(passM, batch * fftN, fftM, static_cast<int32_t>(radixListMHost.size()), isInverse,
                              static_cast<int32_t>(fftN), 1);
        if (!launchMixedPass(kernelM, tilingMDeviceAddrs, input, intermediate, twMatrixM, radixListM, dftMatrixM,
                             passM)) {
            workspace.recycleLast();
            return;
        }
    } else {
        int32_t cntStagesM = Log2PowerOfTwo(fftM);
        if (tilingMDeviceAddrs.size() < static_cast<size_t>(cntStagesM)) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 M radix-2 stage tiling buffers are not initialized.";
            workspace.recycleLast();
            return;
        }
        bool firstMDstScratch = (cntStagesM % 2 == 0);
        for (int32_t stage = 0; stage < cntStagesM; ++stage) {
            bool isLast = (stage == cntStagesM - 1);
            void* stageSrc = nullptr;
            void* stageDst = nullptr;
            if (stage == 0) {
                stageSrc = input;
            } else {
                bool prevDstScratch = firstMDstScratch ? (((stage - 1) & 1) == 0) : (((stage - 1) & 1) != 0);
                stageSrc = prevDstScratch ? static_cast<void*>(scratch) : static_cast<void*>(intermediate);
            }
            if (isLast) {
                stageDst = intermediate;
            } else {
                bool dstScratch = firstMDstScratch ? ((stage & 1) == 0) : ((stage & 1) != 0);
                stageDst = dstScratch ? static_cast<void*>(scratch) : static_cast<void*>(intermediate);
            }

            FftC2CArch35StageTilingData stageTiling;
            FillRadix2StageTiling(stageTiling, batch * fftN, fftM, stage, static_cast<int32_t>(fftN), isInverse,
                                  isLast);
            if (!launchRadix2Stage(kernelM, tilingMDeviceAddrs[stage], stageSrc, stageDst, twMatrixM, radixListM,
                                   dftMatrixM, static_cast<int64_t>(radixListMHost.size()), stageTiling)) {
                workspace.recycleLast();
                return;
            }
        }
    }

    if (useHybrid) {
        // 混合核已直写 output——无 N pass
    } else if (useFusedN) {
        // N 维融合 pass:intermediate [b][ỹ][x] → output [b][x̃][ỹ],级间不落 GM
        if (!launchNFusedPass(kernelNFused, tilingNFusedDeviceAddrs[0], intermediate, output, twNFused)) {
            workspace.recycleLast();
            return;
        }
    } else if (useMixedN) {
        FftAllMixTilingData passN = {};
        FillMixed1DPassTiling(passN, batch * fftM, fftN, static_cast<int32_t>(radixListNHost.size()), isInverse,
                              static_cast<int32_t>(fftM), 1);
        if (!launchMixedPass(kernelN, tilingNDeviceAddrs, intermediate, output, twMatrixN, radixListN, dftMatrixN,
                             passN)) {
            workspace.recycleLast();
            return;
        }
    } else {
        int32_t cntStagesN = Log2PowerOfTwo(fftN);
        if (tilingNDeviceAddrs.size() < static_cast<size_t>(cntStagesN)) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 N radix-2 stage tiling buffers are not initialized.";
            workspace.recycleLast();
            return;
        }
        void* currentN = intermediate;
        for (int32_t stage = 0; stage < cntStagesN; ++stage) {
            bool isLast = (stage == cntStagesN - 1);
            void* stageDst = isLast ?
                                 output :
                                 ((stage & 1) == 0 ? static_cast<void*>(scratch) : static_cast<void*>(intermediate));

            FftC2CArch35StageTilingData stageTiling;
            FillRadix2StageTiling(stageTiling, batch * fftM, fftN, stage, static_cast<int32_t>(fftM), isInverse,
                                  isLast);
            if (!launchRadix2Stage(kernelN, tilingNDeviceAddrs[stage], currentN, stageDst, twMatrixN, radixListN,
                                   dftMatrixN, static_cast<int64_t>(radixListNHost.size()), stageTiling)) {
                workspace.recycleLast();
                return;
            }
            currentN = stageDst;
        }
    }

    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "ASCEND_950 FftC2C2DCoreArch35 run success.";
}

void FftC2C2DCoreArch35::DestroyInDevice()
{
    FreeDeviceAddrs(tilingMDeviceAddrs);
    FreeDeviceAddrs(tilingNDeviceAddrs);
    FreeDeviceAddrs(tilingGemmDeviceAddrs);
    FreeDeviceAddrs(tilingTransDeviceAddrs);
    FreeDeviceAddrs(tilingNFusedDeviceAddrs);
    FreeDeviceAddrs(tilingHybridDeviceAddrs);
}

// GEMM 短维 pass 准入,默认关;档位与开关(ASDSIP_DENSE_GEMM_PASS)
static bool ShouldUseDenseGemmPass(int64_t fftLen, int64_t mTotal)
{
    static const int switchLevel = []() {
        const char* env = std::getenv("ASDSIP_DENSE_GEMM_PASS");
        return env != nullptr ? std::atoi(env) : 0;
    }();
    (void)mTotal; // A/B 实测:batch=4 亏损 4us 可忽略,无需下限
    if (switchLevel <= 0) {
        return false;
    }
    // 上限 64:2L>128 的多 N-tile 路径有 epilogue 缺陷(正确复数列=128-L),修复后再开高档
    return fftLen <= 64;
}

// N 维融合 pass 准入:开关(默认关)+ UB 瓦片预算 + 内核实例覆盖表;扩实例=新内核模板特化+加表项
static bool ShouldUseFusedNPass(int64_t fftN, int64_t fftM)
{
    static const int switchLevel = []() {
        const char* env = std::getenv("ASDSIP_FUSED_NPASS");
        return env != nullptr ? std::atoi(env) : 0;
    }();
    if (switchLevel <= 0) {
        return false;
    }
    if (fftM < 1 || fftM > 52) {
        return false; // UB 瓦片预算(52×1000×4B+tw)
    }
    return fftN == 500; // 实例 (20,25):内核模板按 500 硬编码蝴蝶结构与 ex89 twiddle 布局
}

// 混合核准入:开关 ASDSIP_HYBRID_PASS(默认关)。短维准入同 GEMM(2L≤128 单 N-tile)
// UB 瓦片预算在 tiling 侧复核;混合核替代 GEMM pass,下游 TRANS/NFused 不变
static bool ShouldUseHybridPass(int64_t fftLen, int64_t tileRows)
{
    static const int switchLevel = []() {
        const char* env = std::getenv("ASDSIP_HYBRID_PASS");
        return env != nullptr ? std::atoi(env) : 0;
    }();
    if (switchLevel <= 0) {
        return false;
    }
    if (fftLen <= 0 || fftLen > 64) {
        return false; // 2L≤128:单 N-tile(GEMM 同款准入)
    }
    if (fftLen != 50 || tileRows != 500) {
        return false; // AIV 融合变换体为 500×50 实例硬编码(radix-20×25),他形状走旧路径
    }
    const int64_t nDim = (fftLen * 2 + 7) & ~7;            // UB 行距 32B 对齐(Fixpipe 实证)
    return tileRows * nDim * 4 + 1040 * 4 <= 216LL * 1024; // 瓦片 + 预留 twiddle
}

AspbStatus FftC2C2DCoreArch35::BuildFftPlan()
{
    radixListNHost = FactorizeMixed(fftN);
    radixListMHost = FactorizeMixed(fftM);
    useMixedM = ShouldUseMixedRadixKernel(fftM, static_cast<int64_t>(problemDesc.batch) * fftN);
    useGemmM = ShouldUseDenseGemmPass(fftM, static_cast<int64_t>(problemDesc.batch) * fftN);
    useMixedN = ShouldUseMixedRadixKernel(fftN, static_cast<int64_t>(problemDesc.batch) * fftM);
    useFusedN = ShouldUseFusedNPass(fftN, fftM);
    useHybrid = ShouldUseHybridPass(fftM, fftN);
    if (useHybrid) {
        useGemmM = false; // 混合核替代 GEMM pass(W 换垫宽布局,核通道 mix)
    }
    // useHybrid 直写 output、无 N pass，useFusedN 场景的 coeff 亦不再构建（issue #356）
    if (useFusedN && !useHybrid) {
        // ex89 双向 twiddle 表(2080 浮点)双向共用,forward 键共享缓存
        AsdSip::CoeffKey twNfusedKey = {coreType, 220U, {fftN}, true};
        twNFused = FFTensorCache::getCoeff(twNfusedKey, []() { return BuildFusedNTwiddleTensor(); });
    }

    auto radixListMLocal = radixListMHost;
    auto radixListNLocal = radixListNHost;

    AsdSip::CoeffKey radixMKey = {coreType, useMixedM ? 211U : 202U, {fftM}, problemDesc.forward};
    AsdSip::CoeffKey radixNKey = {coreType, useMixedN ? 210U : 203U, {fftN}, problemDesc.forward};
    AsdSip::CoeffKey twMKey = {coreType, useMixedM ? 213U : 200U, {fftM}, problemDesc.forward};
    AsdSip::CoeffKey twNKey = {coreType, useMixedN ? 212U : 201U, {fftN}, problemDesc.forward};
    const int64_t wStrideM = useHybrid ? ((fftM * 2 + 7) & ~7) : (fftM * 2); // 混合核 W 垫宽(Fixpipe 行距对齐)
    AsdSip::CoeffKey dftMKey = {
        coreType, useGemmM || useHybrid ? 216U : (useMixedM ? 215U : 204U), {fftM, wStrideM}, problemDesc.forward};
    AsdSip::CoeffKey dftNKey = {coreType, useMixedN ? 214U : 205U, {fftN}, problemDesc.forward};

    radixListM = FFTensorCache::getCoeff(radixMKey, [=]() { return BuildRadixListTensor(radixListMLocal); });
    radixListN = FFTensorCache::getCoeff(radixNKey, [=]() { return BuildRadixListTensor(radixListNLocal); });
    twMatrixM = FFTensorCache::getCoeff(twMKey, [=]() -> AsdSip::FFTensor* {
        return useMixedM ? BuildMixedTwiddleTensor(radixListMLocal, problemDesc.forward) :
                           BuildRadix2TwiddleTensor(fftM, problemDesc.forward);
    });
    twMatrixN = FFTensorCache::getCoeff(twNKey, [=]() -> AsdSip::FFTensor* {
        return useMixedN ? BuildMixedTwiddleTensor(radixListNLocal, problemDesc.forward) :
                           BuildRadix2TwiddleTensor(fftN, problemDesc.forward);
    });
    dftMatrixM = FFTensorCache::getCoeff(dftMKey, [=]() -> AsdSip::FFTensor* {
        return (useGemmM || useHybrid) ?
                   BuildGemmDftTensor(fftM, problemDesc.forward, wStrideM) :
                   (useMixedM ? BuildMixedDftTensor(radixListMLocal, problemDesc.forward) : BuildRadix2DftTensor());
    });
    if (useHybrid) {
        // 混合核 coeff=[W 垫宽 | ex89 双向 tw](AIC GEMM 与 AIV 融合变换共用一张 GM 表)
        AsdSip::CoeffKey hybCoeffKey = {coreType, 217U, {fftM, wStrideM}, problemDesc.forward};
        hybridCoeff = FFTensorCache::getCoeff(hybCoeffKey, [=]() -> AsdSip::FFTensor* {
            return BuildHybridCoeffTensor(fftM, problemDesc.forward, wStrideM);
        });
    }
    dftMatrixN = FFTensorCache::getCoeff(dftNKey, [=]() -> AsdSip::FFTensor* {
        return useMixedN ? BuildMixedDftTensor(radixListNLocal, problemDesc.forward) : BuildRadix2DftTensor();
    });

    ASDSIP_LOG(INFO) << "FftC2C2DCoreArch35 BuildFftPlan success. fftN=" << fftN << ", fftM=" << fftM
                     << ", useMixedN=" << useMixedN << ", useMixedM=" << useMixedM << ", useGemmM=" << useGemmM;
    return AsdSip::ErrorType::ACL_SUCCESS;
}

bool FftC2C2DCoreArch35::PreAllocateInDevice()
{
    if (BuildFftPlan() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    return true;
}

AspbStatus FftC2C2DCoreArch35::InitTactic()
{
    int32_t isInverse = 1 - static_cast<int32_t>(problemDesc.forward);

    auto initPass = [&](bool useMixed, int64_t fftLen, int64_t passBatch, int64_t radixListLen,
                        const std::shared_ptr<AsdSip::FFTensor>& dftTensor,
                        const std::shared_ptr<AsdSip::FFTensor>& twTensor,
                        const std::shared_ptr<AsdSip::FFTensor>& radixTensor, std::unique_ptr<Mki::Kernel>& passKernel,
                        std::vector<uint8_t*>& tilingAddrs) -> AsdSip::AspbStatus {
        Tensor tensorIn;
        Tensor tensorOut;
        tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {passBatch, fftLen}, {}, 0};
        tensorIn.dataSize = static_cast<size_t>(passBatch) * static_cast<size_t>(fftLen) *
                            GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        tensorOut.desc = tensorIn.desc;
        tensorOut.dataSize = tensorIn.dataSize;

        OpParam::FftC2CArch35 param = {fftLen, passBatch, radixListLen, isInverse, useMixed};
        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(param);
        passLaunchParam.AddInTensor(tensorIn);
        passLaunchParam.AddInTensor(*dftTensor);
        passLaunchParam.AddInTensor(*twTensor);
        passLaunchParam.AddInTensor(*radixTensor);
        passLaunchParam.AddOutTensor(tensorOut);

        Operation* op = Ops::Instance().GetOperationByName(opName);
        if (op == nullptr) {
            return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        op->InferShape(passLaunchParam);
        passKernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(passLaunchParam));
        ASDSIP_ECHECK(passKernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        passKernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = passKernel->GetTilingSize(passLaunchParam);
        ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
        passKernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
        // Init 失败被吞后仍会发射内核（issue #352）：与同 lambda 既有 ECHECK 口径一致显式拦截
        auto initStatus = passKernel->Init(passLaunchParam);
        ASDSIP_ECHECK(initStatus.Ok(), "pass kernel init failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        int32_t bufferCount = useMixed ? 1 : Log2PowerOfTwo(fftLen);
        FreeDeviceAddrs(tilingAddrs);
        tilingAddrs.reserve(static_cast<size_t>(bufferCount));
        for (int32_t buffer = 0; buffer < bufferCount; ++buffer) {
            void* tempDevicePtr = nullptr;
            int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
            if (st != MKIRT_SUCCESS) {
                FreeDeviceAddrs(tilingAddrs);
                ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "malloc pass tiling device memory fail";
                return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
            }

            uint8_t* deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
            st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                              MKIRT_MEMCOPY_HOST_TO_DEVICE);
            if (st != MKIRT_SUCCESS) {
                MkiRtMemFreeDevice(deviceLaunchBuffer);
                FreeDeviceAddrs(tilingAddrs);
                ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "copy pass tiling to device fail";
                return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
            }
            tilingAddrs.push_back(deviceLaunchBuffer);
        }
        return AsdSip::ErrorType::ACL_SUCCESS;
    };

    auto initGemm = [&](int64_t gemmLen, int64_t gemmRows, int64_t gemmStride,
                        const std::shared_ptr<AsdSip::FFTensor>& wTensor, std::unique_ptr<Mki::Kernel>& passKernel,
                        std::vector<uint8_t*>& tilingAddrs) -> AsdSip::AspbStatus {
        Tensor tensorIn;
        Tensor tensorW;
        tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {gemmRows, gemmLen}, {}, 0};
        tensorIn.dataSize = static_cast<size_t>(gemmRows) * static_cast<size_t>(gemmLen) *
                            GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        tensorW.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {gemmLen * 2, gemmStride}, {}, 0};
        tensorW.dataSize = static_cast<size_t>(gemmLen * 2) * static_cast<size_t>(gemmStride) * sizeof(float);

        OpParam::FftC2CArch35 param = {gemmLen, gemmRows, 0, isInverse, false, true};
        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(param);
        passLaunchParam.AddInTensor(tensorIn);
        passLaunchParam.AddInTensor(*wTensor);

        Tensor tensorOut;
        tensorOut.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {gemmRows, gemmLen}, {}, 0};
        tensorOut.dataSize = tensorIn.dataSize;
        passLaunchParam.AddOutTensor(tensorOut);

        Operation* op = Ops::Instance().GetOperationByName(opName);
        if (op == nullptr) {
            return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        op->InferShape(passLaunchParam);
        passKernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(passLaunchParam));
        ASDSIP_ECHECK(passKernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        passKernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = passKernel->GetTilingSize(passLaunchParam);
        ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
        passKernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
        auto initStatus = passKernel->Init(passLaunchParam);
        ASDSIP_ECHECK(initStatus.Ok(), "gemm kernel init failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        FreeDeviceAddrs(tilingAddrs);
        void* tempDevicePtr = nullptr;
        int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
        ASDSIP_ECHECK(st == MKIRT_SUCCESS, "malloc gemm tiling device memory fail",
                      AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
        uint8_t* deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
        st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                          MKIRT_MEMCOPY_HOST_TO_DEVICE);
        ASDSIP_ECHECK(st == MKIRT_SUCCESS, "copy gemm tiling to device fail",
                      AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
        tilingAddrs.push_back(deviceLaunchBuffer);
        return AsdSip::ErrorType::ACL_SUCCESS;
    };

    auto initHybrid = [&](int64_t hybridLen, int64_t hybridRows, int64_t hybridTileRows,
                          const std::shared_ptr<AsdSip::FFTensor>& wTensor, std::unique_ptr<Mki::Kernel>& passKernel,
                          std::vector<uint8_t*>& tilingAddrs) -> AsdSip::AspbStatus {
        const int64_t hybridNDim = (hybridLen * 2 + 7) & ~7;
        Tensor tensorIn;
        Tensor tensorW;
        tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {hybridRows, hybridLen}, {}, 0};
        tensorIn.dataSize = static_cast<size_t>(hybridRows) * static_cast<size_t>(hybridLen) *
                            GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        tensorW.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {hybridLen * 2, hybridNDim}, {}, 0};
        tensorW.dataSize = static_cast<size_t>(hybridLen * 2) * static_cast<size_t>(hybridNDim) * sizeof(float);

        // 参数打包:fftN=2D fftM(短维 L),batchSize=rows,radixListLen=2D fftN(瓦片行数)
        OpParam::FftC2CArch35 param = {hybridLen, hybridRows, hybridTileRows, isInverse, false, false, false, true};
        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(param);
        passLaunchParam.AddInTensor(tensorIn);
        passLaunchParam.AddInTensor(*wTensor);

        Tensor tensorOut;
        tensorOut.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {hybridRows, hybridLen}, {}, 0};
        tensorOut.dataSize = tensorIn.dataSize;
        passLaunchParam.AddOutTensor(tensorOut);

        Operation* op = Ops::Instance().GetOperationByName(opName);
        if (op == nullptr) {
            return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        op->InferShape(passLaunchParam);
        passKernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(passLaunchParam));
        ASDSIP_ECHECK(passKernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        passKernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = passKernel->GetTilingSize(passLaunchParam);
        ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
        passKernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
        auto initStatus = passKernel->Init(passLaunchParam);
        ASDSIP_ECHECK(initStatus.Ok(), "hybrid kernel init failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        FreeDeviceAddrs(tilingAddrs);
        void* tempDevicePtr = nullptr;
        int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
        ASDSIP_ECHECK(st == MKIRT_SUCCESS, "malloc hybrid tiling device memory fail",
                      AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
        uint8_t* deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
        st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                          MKIRT_MEMCOPY_HOST_TO_DEVICE);
        ASDSIP_ECHECK(st == MKIRT_SUCCESS, "copy hybrid tiling to device fail",
                      AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
        tilingAddrs.push_back(deviceLaunchBuffer);
        return AsdSip::ErrorType::ACL_SUCCESS;
    };

    auto initTrans = [&](int64_t transN, int64_t transM, int64_t transBatch, std::unique_ptr<Mki::Kernel>& passKernel,
                         std::vector<uint8_t*>& tilingAddrs) -> AsdSip::AspbStatus {
        Tensor tensorIn;
        tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {transBatch * transN, transM}, {}, 0};
        tensorIn.dataSize = static_cast<size_t>(transBatch * transN) * static_cast<size_t>(transM) *
                            GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

        // 参数打包(脚手架约定):fftN=transN,batchSize=transBatch,radixListLen=transM
        OpParam::FftC2CArch35 param = {transN, transBatch, transM, isInverse, false, true};
        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(param);
        passLaunchParam.AddInTensor(tensorIn);

        Tensor tensorOut;
        tensorOut.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {transBatch * transM, transN}, {}, 0};
        tensorOut.dataSize = tensorIn.dataSize;
        passLaunchParam.AddOutTensor(tensorOut);

        Operation* op = Ops::Instance().GetOperationByName(opName);
        if (op == nullptr) {
            return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        op->InferShape(passLaunchParam);
        passKernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(passLaunchParam));
        ASDSIP_ECHECK(passKernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        passKernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = passKernel->GetTilingSize(passLaunchParam);
        ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
        passKernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
        auto initStatus = passKernel->Init(passLaunchParam);
        ASDSIP_ECHECK(initStatus.Ok(), "trans kernel init failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        FreeDeviceAddrs(tilingAddrs);
        void* tempDevicePtr = nullptr;
        int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
        ASDSIP_ECHECK(st == MKIRT_SUCCESS, "malloc trans tiling device memory fail",
                      AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
        uint8_t* deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
        st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                          MKIRT_MEMCOPY_HOST_TO_DEVICE);
        ASDSIP_ECHECK(st == MKIRT_SUCCESS, "copy trans tiling to device fail",
                      AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
        tilingAddrs.push_back(deviceLaunchBuffer);
        return AsdSip::ErrorType::ACL_SUCCESS;
    };

    auto initNFused = [&](int64_t fusedLen, int64_t fusedM, std::unique_ptr<Mki::Kernel>& passKernel,
                          std::vector<uint8_t*>& tilingAddrs,
                          const std::shared_ptr<AsdSip::FFTensor>& twTensor) -> AsdSip::AspbStatus {
        Tensor tensorIn;
        tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch * fusedM, fusedLen}, {}, 0};
        tensorIn.dataSize = static_cast<size_t>(problemDesc.batch) * static_cast<size_t>(fusedM) *
                            static_cast<size_t>(fusedLen) *
                            GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

        // 参数打包(脚手架约定):fftN=fusedLen,batchSize=batch,radixListLen=fusedM
        OpParam::FftC2CArch35 param = {fusedLen, problemDesc.batch, fusedM, isInverse, false, false, true};
        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(param);
        passLaunchParam.AddInTensor(tensorIn);
        passLaunchParam.AddInTensor(*twTensor);

        Tensor tensorOut;
        tensorOut.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch * fusedLen, fusedM}, {}, 0};
        tensorOut.dataSize = tensorIn.dataSize;
        passLaunchParam.AddOutTensor(tensorOut);

        Operation* op = Ops::Instance().GetOperationByName(opName);
        if (op == nullptr) {
            return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        op->InferShape(passLaunchParam);
        passKernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(passLaunchParam));
        ASDSIP_ECHECK(passKernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        passKernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = passKernel->GetTilingSize(passLaunchParam);
        ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
        passKernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
        auto initStatus = passKernel->Init(passLaunchParam);
        ASDSIP_ECHECK(initStatus.Ok(), "nfused kernel init failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        FreeDeviceAddrs(tilingAddrs);
        void* tempDevicePtr = nullptr;
        int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
        ASDSIP_ECHECK(st == MKIRT_SUCCESS, "malloc nfused tiling device memory fail",
                      AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
        uint8_t* deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
        st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                          MKIRT_MEMCOPY_HOST_TO_DEVICE);
        ASDSIP_ECHECK(st == MKIRT_SUCCESS, "copy nfused tiling to device fail",
                      AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
        tilingAddrs.push_back(deviceLaunchBuffer);
        return AsdSip::ErrorType::ACL_SUCCESS;
    };

    // 死初始化治理（issue #356）：按 Run 的实际发射分支初始化，useGemmM/useHybrid 生效时
    // kernelM 不发射（Run 分支短路），不再为其分配设备 tiling 缓冲；该场景下 BuildFftPlan
    // 的 dftMatrixM 是 GEMM 垫宽布局（键 216），与 kernelM 期望的 mixed DFT 布局（键 215）
    // 不符，跳过初始化同时消除布局错配隐患
    AspbStatus status = AsdSip::ErrorType::ACL_SUCCESS;
    if (!useGemmM && !useHybrid) {
        status = initPass(useMixedM, fftM, static_cast<int64_t>(problemDesc.batch) * fftN,
                          static_cast<int64_t>(radixListMHost.size()), dftMatrixM, twMatrixM, radixListM, kernelM,
                          tilingMDeviceAddrs);
        if (status != AsdSip::ErrorType::ACL_SUCCESS) {
            return status;
        }
    }

    // useHybrid 无 N pass、useFusedN 用 kernelNFused 替代 kernelN，两者生效时 kernelN 不发射
    if (!useHybrid && !useFusedN) {
        status = initPass(useMixedN, fftN, static_cast<int64_t>(problemDesc.batch) * fftM,
                          static_cast<int64_t>(radixListNHost.size()), dftMatrixN, twMatrixN, radixListN, kernelN,
                          tilingNDeviceAddrs);
        if (status != AsdSip::ErrorType::ACL_SUCCESS) {
            return status;
        }
    }

    if (useGemmM || useHybrid) {
        if (useGemmM) {
            status = initGemm(fftM, static_cast<int64_t>(problemDesc.batch) * fftN, fftM * 2, dftMatrixM, kernelGemm,
                              tilingGemmDeviceAddrs);
            if (status != AsdSip::ErrorType::ACL_SUCCESS) {
                return status;
            }
            // 转置 pass 仅 GEMM 路径需要；混合核直写 output、无 TRANS，不再初始化 kernelTrans
            status = initTrans(fftN, fftM, static_cast<int64_t>(problemDesc.batch), kernelTrans,
                               tilingTransDeviceAddrs);
            if (status != AsdSip::ErrorType::ACL_SUCCESS) {
                return status;
            }
        } else {
            // 混合核:W 为垫宽布局(RoundUp8(2L));radixListLen 携带 2D fftN(瓦片行数)
            status = initHybrid(fftM, static_cast<int64_t>(problemDesc.batch) * fftN, fftN, dftMatrixM, kernelHybrid,
                                tilingHybridDeviceAddrs);
            if (status != AsdSip::ErrorType::ACL_SUCCESS) {
                return status;
            }
        }
    }

    // useHybrid 直写 output、N pass 一并省略，kernelNFused 同样不发射
    if (useFusedN && !useHybrid) {
        status = initNFused(fftN, fftM, kernelNFused, tilingNFusedDeviceAddrs, twNFused);
        if (status != AsdSip::ErrorType::ACL_SUCCESS) {
            return status;
        }
    }

    ASDSIP_LOG(INFO) << "FftC2C2DCoreArch35 init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
