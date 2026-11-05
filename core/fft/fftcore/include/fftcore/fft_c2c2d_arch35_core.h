/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_C2C2D_ARCH35_CORE__
#define __FFT_C2C2D_ARCH35_CORE__

#include <cstdint>
#include <vector>
#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

class FftC2C2DCoreArch35 : public FFTCoreBase {
public:
    // mode 死成员删除（issue #356）：构造入参从未被核内读取；RADIX2 分支静态不可达（issue #357）
    FftC2C2DCoreArch35(int64_t fftN, int64_t fftM, int64_t batch, AsdSip::asdFftType fftType, bool forward)
        : FFTCoreBase(FFTCoreType::kFftC2C2DArch35, 1, static_cast<unsigned>(fftN), 1, static_cast<unsigned>(batch),
                      fftType, forward),
          fftN(fftN),
          fftM(fftM)
    {}
    ~FftC2C2DCoreArch35() override { DestroyInDevice(); }

    size_t EstimateWorkspaceSize() override;
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override {}
    void Run(void* input, void* output, void* stream, workspace::Workspace& workspace) override;

private:
    bool PreAllocateInDevice() override;
    void DestroyInDevice();
    AsdSip::AspbStatus BuildFftPlan();
    AsdSip::AspbStatus InitTactic();

    int64_t fftN;
    int64_t fftM;
    std::vector<int32_t> radixListNHost;
    std::vector<int32_t> radixListMHost;
    std::shared_ptr<AsdSip::FFTensor> radixListN;
    std::shared_ptr<AsdSip::FFTensor> radixListM;
    std::shared_ptr<AsdSip::FFTensor> twMatrixM;
    std::shared_ptr<AsdSip::FFTensor> twMatrixN;
    std::shared_ptr<AsdSip::FFTensor> dftMatrixM;
    std::shared_ptr<AsdSip::FFTensor> dftMatrixN;
    bool useMixedM{false};
    bool useMixedN{false};
    bool useGemmM{false};
    bool useFusedN{false};
    bool useHybrid{false}; // 混合核(开关 ASDSIP_HYBRID_PASS):替代 GEMM pass,下游 TRANS/NFused 不变
    bool gemmWarmupDone{false};
    bool hybridWarmupDone{false};
    std::unique_ptr<Mki::Kernel> kernelM;
    std::unique_ptr<Mki::Kernel> kernelN;
    std::unique_ptr<Mki::Kernel> kernelGemm;
    std::unique_ptr<Mki::Kernel> kernelTrans;
    std::unique_ptr<Mki::Kernel> kernelNFused;
    std::unique_ptr<Mki::Kernel> kernelHybrid;
    std::vector<uint8_t*> tilingMDeviceAddrs;
    std::vector<uint8_t*> tilingNDeviceAddrs;
    std::vector<uint8_t*> tilingGemmDeviceAddrs;
    std::vector<uint8_t*> tilingTransDeviceAddrs;
    std::vector<uint8_t*> tilingNFusedDeviceAddrs;
    std::vector<uint8_t*> tilingHybridDeviceAddrs;
    std::shared_ptr<AsdSip::FFTensor> twNFused;
    std::shared_ptr<AsdSip::FFTensor> hybridCoeff; // [W 垫宽 | ex89 双向 tw] 拼接 coeff
    std::string opName{"FftC2CArch35Operation"};
};

#endif
