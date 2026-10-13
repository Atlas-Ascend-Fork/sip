/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_C2R_ARCH35_CORE__
#define __FFT_C2R_ARCH35_CORE__

#include <vector>
#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

/**
 * Single stage of the FFT execution plan, corresponding to one entry in
 * Python build_fft_plan's plan list:
 *   plan.append({'radix': radix, 'M': M, 'W_R': W_R, 'T': T})
 *
 * W_R: DFT/IDFT matrix of size [radix, radix] (complex, stored as 2*radix*radix floats)
 * T:   Twiddle factor matrix of size [radix, M] (complex, stored as 2*radix*M floats)
 */
struct FftC2RPlanStage {
    int64_t radix; // current stage radix, must be one of ALLOWED_RADICES
    int64_t M;     // temp_N / radix
};

class FftC2RCoreArch35 : public FFTCoreBase {
public:
    FftC2RCoreArch35(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
                     bool forward)
        : FFTCoreBase(FFTCoreType::kFftC2RArch35, nDone, nDoing, nLeft, batch, fftType, forward),
          parity((nDoing % 2 != 0) ? 1 : 0)
    // NO hardware optimization: use full nDoing for decomposition
    // Matches fft_c2r_test.py with is_c2r=False
    {}
    ~FftC2RCoreArch35() override { DestroyInDevice(); }
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override {}
    void Run(void* input, void* output, void* stream, workspace::Workspace& workspace) override;
    // ops-fft 后端已覆盖 950/arch35 混合基路径 (fft_c2r_multi_core 直调)
    bool OpsFftBackendAdapted() const override { return true; }
    bool OpsFftBackendStubReady() const override;

    // Supported prime radices for mixed-radix decomposition.
    // Extend this array to support additional radices (e.g., 11, 13).
    static constexpr int64_t ALLOWED_RADICES[] = {2, 3, 5, 7};

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice();

    /**
     * Build the FFT execution plan (strictly matches fft_c2r_test.py build_fft_plan).
     * Decomposes nDoing into prime factors from ALLOWED_RADICES and constructs
     * W_R and T matrices for each stage.
     *
     * NO hardware optimization: uses full nDoing (not nDoing/2)
     * Matches Python: build_fft_plan(N, is_forward=True, is_c2r=False)
     */
    AsdSip::AspbStatus BuildFftPlan();

    AsdSip::AspbStatus InitTactic();

    // ops-fft 直调后端 (950/arch35 混合基)
    void RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace);

    int parity{0};
    // NO nFftC2c member - use full nDoing for decomposition (no hardware optimization)

    // The execution plan: one entry per decomposition stage
    std::vector<FftC2RPlanStage> plan;

    // Packed tensors for kernel consumption
    std::shared_ptr<AsdSip::FFTensor> radixListTensor; // [planLen] int32, the radix of each stage
    std::shared_ptr<AsdSip::FFTensor> dftMatrixArray;  // concatenated W_R matrices (float, real-imag interleaved)
    std::shared_ptr<AsdSip::FFTensor> twMatrixArray;   // concatenated T matrices (float, real-imag interleaved)

    // 常量 device 缓存 (plan 级)
    void* opsDftMatrix = nullptr;
    void* opsTwMatrix = nullptr;
    void* opsRadixList = nullptr;
    uint32_t opsCachedBlocks = 0;

    std::string opName{"FftC2ROperation"};
};

#endif
