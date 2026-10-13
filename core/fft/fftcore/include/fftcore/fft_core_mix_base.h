/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_CORE_MIX_BASE__
#define __FFT_CORE_MIX_BASE__

#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

class FftCoreMixBase : public FFTCoreBase {
public:
    FftCoreMixBase(FFTCoreType coreType, unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch,
                   AsdSip::asdFftType fftType, bool forward, int parity, unsigned nFftC2c)
        : FFTCoreBase(coreType, nDone, nDoing, nLeft, batch, fftType, forward), parity{parity}, nFftC2c{nFftC2c}
    {}
    ~FftCoreMixBase() override { DestroyInDevice(); }
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override;
    void Run(void* input, void* output, void* stream, workspace::Workspace& workspace) override;
    // ops-fft 后端适配: 混合基族 (kFftMix/kFftR2C/kFftC2R) 均可直调 ops-fft kernel
    bool OpsFftBackendAdapted() const override { return true; }
    bool OpsFftBackendStubReady() const override;

protected:
    void InitRadix() override;
    AsdSip::AspbStatus InitTactic();
    bool PreAllocateC2CInDevice();
    bool PreAllocateX2XInDevice();
    void DestroyInDevice();

    virtual void InitParams();
    virtual std::string GetOpName() = 0;
    bool PreAllocateInDevice() override { return true; }
    virtual void InitLaunchParam() = 0;

    int aivSplitWay{0};
    int parity{0};
    unsigned nFftC2c{0};

    int64_t workspaceInputSize{0};
    int64_t workspaceOutputSize{0};
    int64_t workspaceSyncSize{0};
    int64_t workspaceC2cOutputSize{0};
    int64_t workspaceAuxilSize{0};

    std::shared_ptr<AsdSip::FFTensor> radixList;
    std::shared_ptr<AsdSip::FFTensor> dftMatrixArray;
    std::shared_ptr<AsdSip::FFTensor> twMatrixArray;
    std::shared_ptr<AsdSip::FFTensor> inputIndex;
    std::shared_ptr<AsdSip::FFTensor> outputIndex;
    std::shared_ptr<AsdSip::FFTensor> aTable;
    std::shared_ptr<AsdSip::FFTensor> bTable;

    Mki::Any opParam;

private:
    // ops-fft 直调后端
    void RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace);
    // 常量 device 缓存 (plan 级)
    void* opsDftMatrix = nullptr;
    void* opsTwMatrix = nullptr;
    void* opsRadixList = nullptr;
    void* opsInputIndex = nullptr;
    void* opsOutputIndex = nullptr;
    void* opsATable = nullptr;
    void* opsBTable = nullptr;
    uint32_t opsCachedBlocks = 0;
    bool opsConstUploaded = false;
};

#endif
