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
#include "params/fft_transpose.h"
#include "tiling.h"
#include "utils/assert.h"

namespace AsdSip {
using namespace Mki;

class FftTransposeOperation : public OperationBase {
public:
    explicit FftTransposeOperation(const std::string& name) noexcept : OperationBase(name) {}
    int64_t GetInputNum(const Any& p) const override { return p.Type() == typeid(OpParam::FftTranspose) ? 1 : 0; }
    Kernel* GetBestKernel(const LaunchParam& p) const override
    {
        return IsConsistent(p) ? GetKernelByName("FftTransposeKernel") : nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launch, SVector<Tensor>& outputs) const override
    {
        ASDSIP_CHECK(launch.GetParam().Type() == typeid(OpParam::FftTranspose) && launch.GetInTensorCount() == 1 &&
                         outputs.size() == 1,
                     "Invalid transpose arguments", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        const auto& p = AnyCast<OpParam::FftTranspose>(launch.GetParam());
        ASDSIP_CHECK(p.Valid(), "Invalid transpose shape", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        outputs[0].desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {p.batch, p.cols, p.rows}, {}, 0};
        return Status::OkStatus();
    }
};

class FftTransposeKernel : public KernelBase {
public:
    FftTransposeKernel(const std::string& name, const BinHandle* handle) noexcept : KernelBase(name, handle) {}
    bool CanSupport(const LaunchParam& launch) const override
    {
        if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950 ||
            launch.GetParam().Type() != typeid(OpParam::FftTranspose) || launch.GetInTensorCount() != 1 ||
            launch.GetOutTensorCount() != 1) {
            return false;
        }
        const auto& p = AnyCast<OpParam::FftTranspose>(launch.GetParam());
        if (!p.Valid()) {
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            const auto& t = i ? launch.GetOutTensor(0) : launch.GetInTensor(0);
            const SVector<int64_t> shape = i ? SVector<int64_t>{p.batch, p.cols, p.rows} :
                                               SVector<int64_t>{p.batch, p.rows, p.cols};
            if (t.desc.dtype != TENSOR_DTYPE_COMPLEX64 || !t.IsContiguous() || t.desc.offset != 0 ||
                t.desc.dims != shape) {
                return false;
            }
        }
        return true;
    }
    uint64_t GetTilingSize(const LaunchParam&) const override { return sizeof(FftTransposeTiling); }
    Status InitImpl(const LaunchParam& launch) override
    {
        ASDSIP_CHECK(CanSupport(launch), "Unsupported transpose", return Status::FailStatus(ERROR_INVALID_VALUE));
        const auto& p = AnyCast<OpParam::FftTranspose>(launch.GetParam());
        const int64_t cores = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
        ASDSIP_CHECK(cores > 0 && kernelInfo_.GetTilingHostAddr(), "Missing AIV/tiling",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        constexpr int tile = 64;
        constexpr int slots = 2;
        constexpr int64_t ub = slots * 2 * tile * (tile + 4) * 8;
        ASDSIP_CHECK(ub + 4096 <= static_cast<int64_t>(PlatformInfo::Instance().GetUbSize()),
                     "Transpose exceeds UB budget", return Status::FailStatus(ERROR_INVALID_VALUE));
        *reinterpret_cast<FftTransposeTiling*>(kernelInfo_.GetTilingHostAddr()) = {p.batch, p.rows, p.cols};
        const int64_t work = p.batch * ((p.rows + tile - 1) / tile) * ((p.cols + tile - 1) / tile);
        kernelInfo_.SetBlockDim(static_cast<uint32_t>(std::min(cores, work)));
        kernelInfo_.SetTilingId(1);
        kernelInfo_.SetLocalMemorySize(0);
        kernelInfo_.GetScratchSizes() = {0};
        return Status::OkStatus();
    }
};
REG_OPERATION(FftTransposeOperation);
REG_KERNEL_BASE(FftTransposeKernel);
} // namespace AsdSip
