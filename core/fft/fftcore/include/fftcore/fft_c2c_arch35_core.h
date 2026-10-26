/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_C2C_ARCH35_CORE__
#define __FFT_C2C_ARCH35_CORE__

#include <cstdint>
#include <vector>
#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

struct FftC2CPlanStage {
    int64_t radix;
    int64_t M;
};

class FftC2CCoreArch35 : public FFTCoreBase {
public:
    FftC2CCoreArch35(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
                     bool forward)
        : FFTCoreBase(FFTCoreType::kFftC2CArch35, nDone, nDoing, nLeft, batch, fftType, forward)
    {}
    ~FftC2CCoreArch35() override { DestroyInDevice(); }
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override {}
    void Run(void* input, void* output, void* stream, workspace::Workspace& workspace) override;
    // ops-fft 后端已覆盖 950 两条路径: mixed-radix 直调 + radix-2 直调制品旧版 kernel
    bool OpsFftBackendAdapted() const override { return true; }
    bool OpsFftBackendStubReady() const override;

    static constexpr int64_t ALLOWED_RADICES[] = {2, 3, 5, 7, 11, 13, 17, 19};

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice();
    AsdSip::AspbStatus BuildFftPlan();
    AsdSip::AspbStatus InitTactic();
    // ops-fft 直调后端 (950/arch35):
    // mixed-radix 经 mix 内核整 FFT 单 launch 直调; radix-2 纯 2 幂直调制品旧版 kernel
    // (即 sip 97769d8 优化前实现); 常量拷贝为公共步骤
    void RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace);
    void RunViaOpsFftRadix2(void* input, void* output, void* stream, workspace::Workspace& workspace);
    bool UploadOpsFftConstants();
    // radix-2 直调常量: per-stage DFT 矩阵 + 收缩式 twiddle 级联 (旧版契约, 数值等价判据)
    AsdSip::AspbStatus BuildOpsFftRadix2Coeffs();

    std::vector<FftC2CPlanStage> plan;
    bool isMixedRadix{false};
    std::shared_ptr<AsdSip::FFTensor> radixListTensor;
    std::shared_ptr<AsdSip::FFTensor> dftMatrixArray;
    std::shared_ptr<AsdSip::FFTensor> twMatrixArray;
    std::vector<uint8_t*> stageTilingDeviceAddrs;
    // 常量 device 缓存 (plan 级)
    void* opsDftMatrix = nullptr;
    void* opsTwMatrix = nullptr;
    void* opsRadixList = nullptr;
    // ops-fft radix-2 路径: 旧版契约常量 host 侧缓存 (FFTensorCache plan 级) 与 tiling device 缓存
    std::shared_ptr<AsdSip::FFTensor> opsRadix2DftMatrix;
    std::shared_ptr<AsdSip::FFTensor> opsRadix2TwMatrix;
    std::shared_ptr<AsdSip::FFTensor> opsRadix2RadixList;
    void* opsTilingDeviceAddr = nullptr;
    uint32_t opsCachedBlocks = 0;

    std::string opName{"FftC2CArch35Operation"};
};

#endif
