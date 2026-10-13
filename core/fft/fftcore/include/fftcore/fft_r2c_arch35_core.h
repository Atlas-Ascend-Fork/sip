/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_R2C_ARCH35_CORE__
#define __FFT_R2C_ARCH35_CORE__

#include <vector>
#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

/**
 * Single stage of the FFT execution plan for R2C.
 * Corresponds to one entry in Python build_fft_plan's plan list:
 *   plan.append({'radix': radix, 'M': M, 'W_R': W_R, 'T': T})
 *
 * W_R: DFT matrix of size [radix, radix] (complex, stored as 2*radix*radix floats)
 * T:   Twiddle factor matrix of size [radix, M] (complex, stored as 2*radix*M floats)
 */
struct FftR2CPlanStage {
    int64_t radix;
    int64_t M;
};

class FftR2CCoreArch35 : public FFTCoreBase {
public:
    FftR2CCoreArch35(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
                     bool forward)
        : FFTCoreBase(FFTCoreType::kFftR2CArch35, nDone, nDoing, nLeft, batch, fftType, forward)
    {}
    ~FftR2CCoreArch35() override { DestroyInDevice(); }
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override {}
    void Run(void* input, void* output, void* stream, workspace::Workspace& workspace) override;
    // ops-fft 后端已覆盖 950/arch35 混合基路径 (fft_r2c_multi_core 直调)
    bool OpsFftBackendAdapted() const override { return true; }
    bool OpsFftBackendStubReady() const override;

    static constexpr int64_t ALLOWED_RADICES[] = {2, 3, 5, 7};

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice();

    /**
     * Build the FFT execution plan for R2C.
     * Decomposes N/2 into prime factors from ALLOWED_RADICES and constructs
     * W_R and T matrices for each stage.
     * Also builds post-process twiddle tensor W_N^k for Phase 3.
     *
     * R2C uses N/2 for decomposition (packed complex FFT).
     * Matches Python: build_fft_plan(N, is_forward=True, is_r2c=True)
     */
    AsdSip::AspbStatus BuildFftPlan();

    AsdSip::AspbStatus InitTactic();

    // ops-fft 直调后端 (950/arch35 混合基)
    void RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace);

    std::vector<FftR2CPlanStage> plan;

    std::shared_ptr<AsdSip::FFTensor> radixListTensor;
    std::shared_ptr<AsdSip::FFTensor> dftMatrixArray;
    std::shared_ptr<AsdSip::FFTensor> twMatrixArray;
    std::shared_ptr<AsdSip::FFTensor> twPostProcess;

    // 常量 device 缓存 (plan 级)
    void* opsDftMatrix = nullptr;
    void* opsTwMatrix = nullptr;
    void* opsTwPostProcess = nullptr;
    void* opsRadixList = nullptr;
    uint32_t opsCachedBlocks = 0;

    std::string opName{"FftR2COperation"};
};

#endif
