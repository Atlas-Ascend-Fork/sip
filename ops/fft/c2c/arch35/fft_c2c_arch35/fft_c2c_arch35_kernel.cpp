/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/math/math.h>
#include <mki/utils/rt/module/module.h>
#include <mki_loader/op_register.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "fft_all_mix.h"
#include "fft_c2c_arch35.h"
#include "../../../common/include/tiling/fft_all_mix_tiling_data.h"
#include "tiling/fft_c2c_arch35_tiling.h"
#include "tiling/fft_c2c_arch35_gemm_tiling.h"

namespace AsdSip {
constexpr uint64_t TENSOR_PERM_IDX = 1;

namespace {
bool IsPowerOfTwo(int64_t n) { return n > 1 && (n & (n - 1)) == 0; }

bool HasSupportedFactors(int64_t n)
{
    if (n <= 1) {
        return false;
    }
    constexpr int64_t allowedRadices[] = {2, 3, 5, 7, 11, 13, 17, 19};
    for (int64_t radix : allowedRadices) {
        while (n % radix == 0) {
            n /= radix;
        }
    }
    return n == 1;
}
} // namespace

class FftC2CArch35Kernel : public KernelBase {
public:
    explicit FftC2CArch35Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : KernelBase(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CArch35),
                         "FftC2CArch35: param type invalid", return false);
            ASDSIP_CHECK(launchParam.GetInTensorCount() == 4, "input num invalid", return false);
            ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        }
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        uint64_t c2cTilingSize = static_cast<uint64_t>(sizeof(FftAllMixTilingData));
        return c2cTilingSize;
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = FftC2CArch35Tiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftC2CArch35Tiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }

protected:
    // UB 窗口必须随发射下发(mki KernelBase 丢运行时配置):localMemory=0 回落 KernelBase 默认发射;
    // 供 NFused/Hybrid 等 override 复用,tag 用于错误消息定位
    Status LaunchWithLocalMemory(const std::string& tag, void* args, void* stream, bool isDeviceAddr)
    {
        if (kernelInfo_.GetLocalMemorySize() == 0) {
            return KernelBase::RunWithArgs(args, stream, isDeviceAddr);
        }
        constexpr uint32_t argumentCount = 5; // 2 入 + 1 出 + tiling + overflow
        constexpr uint32_t tilingArgument = 3;
        const auto* binary = GetBinHandle();
        ASDSIP_CHECK(binary != nullptr && args != nullptr && kernelInfo_.GetConstTensorCount() == 0 &&
                         kernelInfo_.GetArgsSize() >= argumentCount * sizeof(void*) &&
                         kernelInfo_.GetArgsSize() <= UINT32_MAX,
                     "Invalid " + tag + " runtime arguments", return Status::FailStatus(ERROR_INVALID_VALUE));
        RtArgsExT argsEx{};
        argsEx.args = args;
        argsEx.argsSize = static_cast<uint32_t>(kernelInfo_.GetArgsSize());
        argsEx.isNoNeedH2DCopy = isDeviceAddr ? 1 : 0;
        if (kernelInfo_.GetLaunchWithTiling()) {
            argsEx.hasTiling = 1;
            argsEx.tilingAddrOffset = tilingArgument * sizeof(void*);
            argsEx.tilingDataOffset = argumentCount * sizeof(void*);
        }
        MkiRtKernelParam launch{};
        launch.tilingId = kernelInfo_.GetTilingId();
        launch.blockDim = kernelInfo_.GetBlockDim();
        launch.argsEx = &argsEx;
        RtTaskCfgInfoT config{};
        config.localMemorySize = kernelInfo_.GetLocalMemorySize();
        config.schemMode = static_cast<uint8_t>(kernelInfo_.GetScheduleMode());
        const auto handle = binary->GetHandle();
        const int result = *handle != nullptr ? MkiRtFunctionLaunchWithHandle(*handle, &launch, stream, &config) :
                                                MkiRtFunctionLaunchWithFlag(handle, &launch, stream, &config);
        ASDSIP_CHECK(result == MKIRT_SUCCESS, tag + " launch with local memory failed: " + std::to_string(result),
                     return Status::FailStatus(ERROR_LAUNCH_KERNEL_ERROR));
        return Status::OkStatus();
    }
};

class FftC2CArch35C64Kernel : public FftC2CArch35Kernel {
public:
    explicit FftC2CArch35C64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : FftC2CArch35Kernel(kernelName, handle)
    {}

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return static_cast<uint64_t>(sizeof(FftC2CArch35StageTilingData));
    }

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(FftC2CArch35Kernel::CanSupport(launchParam), "FftC2CArch35C64Kernel failed to check support",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                     return false);
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        ASDSIP_CHECK(!param.isMixedRadix, "FftC2CArch35C64Kernel requires radix-2 mode", return false);
        ASDSIP_CHECK(IsPowerOfTwo(param.fftN), "FftC2CArch35 radix-2 kernel requires power-of-two fftN", return false);
        return true;
    }
};

REG_KERNEL_BASE(FftC2CArch35C64Kernel);

class FftC2CArch35MixC64Kernel : public FftC2CArch35Kernel {
public:
    explicit FftC2CArch35MixC64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : FftC2CArch35Kernel(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(FftC2CArch35Kernel::CanSupport(launchParam), "FftC2CArch35MixC64Kernel failed to check support",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT, "dft matrix dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(2).desc.dtype == TENSOR_DTYPE_FLOAT, "twiddle matrix dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(3).desc.dtype == TENSOR_DTYPE_INT32, "radix list dtype unsupported",
                     return false);
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        ASDSIP_CHECK(param.isMixedRadix, "FftC2CArch35MixC64Kernel requires mixed-radix mode", return false);
        ASDSIP_CHECK(HasSupportedFactors(param.fftN), "FftC2CArch35 mixed kernel requires supported factors",
                     return false);
        return true;
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = FftC2CArch35MixTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftC2CArch35MixTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

REG_KERNEL_BASE(FftC2CArch35MixC64Kernel);

class FftC2CArch35GemmC64Kernel : public FftC2CArch35Kernel {
public:
    explicit FftC2CArch35GemmC64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : FftC2CArch35Kernel(kernelName, handle)
    {}

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return static_cast<uint64_t>(sizeof(FftGemmTilingData));
    }

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() != Mki::PlatformType::ASCEND_950) {
            return false;
        }
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CArch35),
                     "FftC2CArch35GemmC64Kernel: param type invalid", return false);
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        ASDSIP_CHECK(param.isGemm, "FftC2CArch35GemmC64Kernel requires gemm mode", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2, "gemm input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "input dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT, "W dtype unsupported", return false);
        return true;
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = FftC2CArch35GemmTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftC2CArch35GemmTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

REG_KERNEL_BASE(FftC2CArch35GemmC64Kernel);

class FftC2CArch35TransC64Kernel : public FftC2CArch35Kernel {
public:
    explicit FftC2CArch35TransC64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : FftC2CArch35Kernel(kernelName, handle)
    {}

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return static_cast<uint64_t>(sizeof(FftTransTilingData));
    }

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() != Mki::PlatformType::ASCEND_950) {
            return false;
        }
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CArch35),
                     "FftC2CArch35TransC64Kernel: param type invalid", return false);
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        ASDSIP_CHECK(param.isGemm, "FftC2CArch35TransC64Kernel requires gemm mode", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 1, "trans input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "input dtype unsupported",
                     return false);
        return true;
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = FftC2CArch35TransTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftC2CArch35TransTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

REG_KERNEL_BASE(FftC2CArch35TransC64Kernel);

class FftC2CArch35NFusedC64Kernel : public FftC2CArch35Kernel {
public:
    explicit FftC2CArch35NFusedC64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : FftC2CArch35Kernel(kernelName, handle)
    {}

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return static_cast<uint64_t>(sizeof(FftNFusedTilingData));
    }

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() != Mki::PlatformType::ASCEND_950) {
            return false;
        }
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CArch35),
                     "FftC2CArch35NFusedC64Kernel: param type invalid", return false);
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        ASDSIP_CHECK(param.isNFused, "FftC2CArch35NFusedC64Kernel requires nfused mode", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2, "nfused input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "input dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT, "twiddle dtype unsupported",
                     return false);
        return true;
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = FftC2CArch35NFusedTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftC2CArch35NFusedTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }

    // UB 窗口必须随发射下发:MKI KernelBase 丢运行时配置,复制 c2c large 族 override
    Status RunWithArgs(void* args, void* stream, bool isDeviceAddr) override
    {
        return LaunchWithLocalMemory("nfused", args, stream, isDeviceAddr);
    }
};

REG_KERNEL_BASE(FftC2CArch35NFusedC64Kernel);

class FftC2CArch35HybridC64Kernel : public FftC2CArch35Kernel {
public:
    explicit FftC2CArch35HybridC64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : FftC2CArch35Kernel(kernelName, handle)
    {}

    uint64_t GetTilingSize(const LaunchParam& launchParam) const override
    {
        (void)launchParam;
        return static_cast<uint64_t>(sizeof(FftHybridTilingData));
    }

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() != Mki::PlatformType::ASCEND_950) {
            return false;
        }
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CArch35),
                     "FftC2CArch35HybridC64Kernel: param type invalid", return false);
        const auto& param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        ASDSIP_CHECK(param.isHybrid, "FftC2CArch35HybridC64Kernel requires hybrid mode", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2, "hybrid input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "input dtype unsupported",
                     return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT, "W dtype unsupported", return false);
        return true;
    }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        auto status = FftC2CArch35HybridTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftC2CArch35HybridTiling failed",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }

    // UB 窗口必须随发射下发(mix 核同样适用;复制 nfused override,2 入 + 1 出 + tiling + overflow 同构)
    Status RunWithArgs(void* args, void* stream, bool isDeviceAddr) override
    {
        return LaunchWithLocalMemory("hybrid", args, stream, isDeviceAddr);
    }
};

REG_KERNEL_BASE(FftC2CArch35HybridC64Kernel);
} // namespace AsdSip
