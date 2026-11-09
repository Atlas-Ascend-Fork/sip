/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <mki/base/operation_base.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>
#include "params/fft_mixed_common.h"
#include "utils/assert.h"

namespace AsdSip {
using namespace Mki;
using namespace OpParam;

class FftMixedCommonOperation : public OperationBase {
public:
    explicit FftMixedCommonOperation(const std::string& name) noexcept : OperationBase(name) {}
    int64_t GetInputNum(const Any& p) const override { return p.Type() == typeid(FftMixedCommon) ? 2 : 0; }
    Kernel* GetBestKernel(const LaunchParam& p) const override
    {
        return IsConsistent(p) && p.GetParam().Type() == typeid(FftMixedCommon) ?
                   GetKernelByName("FftMixedCommonKernel") :
                   nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launch, SVector<Tensor>& outputs) const override
    {
        ASDSIP_CHECK(
            launch.GetParam().Type() == typeid(FftMixedCommon) && launch.GetInTensorCount() == 2 && outputs.size() == 1,
            "Invalid FFT mixed common arguments", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        const auto& p = AnyCast<FftMixedCommon>(launch.GetParam());
        ASDSIP_CHECK(ValidMixedCommon(p), "Invalid FFT mixed common shape/pass",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        outputs[0].desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {MixedCommonFloats(p, true)}, {}, 0};
        return Status::OkStatus();
    }
};

class FftMixedCommonKernel : public KernelBase {
public:
    FftMixedCommonKernel(const std::string& name, const BinHandle* handle) noexcept : KernelBase(name, handle) {}
    bool CanSupport(const LaunchParam& launch) const override
    {
        if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950 ||
            launch.GetParam().Type() != typeid(FftMixedCommon) || launch.GetInTensorCount() != 2 ||
            launch.GetOutTensorCount() != 1)
            return false;
        const auto& p = AnyCast<FftMixedCommon>(launch.GetParam());
        if (!ValidMixedCommon(p))
            return false;
        for (int i = 0; i < 3; ++i) {
            const auto& t = i == 2 ? launch.GetOutTensor(0) : launch.GetInTensor(i);
            const int64_t count = i == 1 ? MixedCommonCoeffFloats(p) : MixedCommonFloats(p, i == 2);
            if (count < 0)
                return false;
            if (t.desc.dtype != TENSOR_DTYPE_FLOAT || !t.IsContiguous() || t.desc.offset != 0 ||
                t.desc.dims.size() != 1 || t.desc.dims[0] != count)
                return false;
        }
        return true;
    }
    uint64_t GetTilingSize(const LaunchParam&) const override { return sizeof(MixedCommonTiling); }
    Status InitImpl(const LaunchParam& launch) override
    {
        ASDSIP_CHECK(CanSupport(launch), "Unsupported FFT mixed common pass",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        const auto& p = AnyCast<FftMixedCommon>(launch.GetParam());
        const int64_t cores = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
        ASDSIP_CHECK(cores > 0 && kernelInfo_.GetTilingHostAddr(), "Missing AIV/tiling",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        auto& tiling = *reinterpret_cast<MixedCommonTiling*>(kernelInfo_.GetTilingHostAddr());
        ASDSIP_CHECK(BuildMixedCommonTiling(p, tiling), "Invalid mixed common factor plan",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetBlockDim(std::min(cores, p.rows));
        kernelInfo_.SetTilingId(p.n <= 128 ? 0 : p.n <= 512 ? 1 : p.n <= 2048 ? 2 : 3);
        kernelInfo_.SetLocalMemorySize(0);
        kernelInfo_.GetScratchSizes() = {0};
        return Status::OkStatus();
    }
};
REG_OPERATION(FftMixedCommonOperation);
REG_KERNEL_BASE(FftMixedCommonKernel);
} // namespace AsdSip
