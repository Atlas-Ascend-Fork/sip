/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __DFT_R2C_CORE__
#define __DFT_R2C_CORE__

#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

class DftR2CCore : public FFTCoreBase {
public:
    DftR2CCore(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
               bool forward)
        : FFTCoreBase(FFTCoreType::kDftR2C, nDone, nDoing, nLeft, batch, fftType, forward)
    {}
    ~DftR2CCore() override { DestroyInDevice(); }
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override {}
    void Run(void* input, void* output, void* stream, workspace::Workspace& workspace) override;
    bool OpsFftBackendAdapted() const override { return true; }
    bool OpsFftBackendStubReady() const override;

private:
    std::string opName{"DftR2COperation"};
    std::shared_ptr<AsdSip::FFTensor> rotationMatrix;

    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice();
    AsdSip::AspbStatus InitRotationMatrix();
    AsdSip::AspbStatus InitTactic();
    // ops-fft 直调后端: dft_r2c 的 tiling 为指针传递
    void RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace);

    // 常量 device 缓存 (plan 级)
    void* opsRotationMatrix = nullptr;
    uint32_t opsCachedBlocks = 0;
};

#endif
