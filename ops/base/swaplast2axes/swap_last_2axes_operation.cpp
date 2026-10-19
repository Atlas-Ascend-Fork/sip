/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mki/base/operation_base.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mki_loader/op_register.h"
#include "mki/utils/SVector/SVector.h"
#include "swap_last_2axes.h"

namespace AsdSip {
using namespace Mki;

constexpr int64_t INPUT_DIM = 1;

class SwapLast2AxesOperation : public OperationBase {
public:
    explicit SwapLast2AxesOperation(const std::string& opName) noexcept : OperationBase(opName) {}

    Kernel* GetBestKernel(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);

        // F32 分支修正（issue #255）：全仓无 SwapLast2AxesF32Kernel 实现/注册
        // （CMakeLists 仅构建 C64、op_list.yaml 仅登记 C64），原返回 nullptr 使
        // float32 输入派核失败且分支形同死代码；显式报告不支持的数据类型
        if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT) {
            ASDSIP_LOG(ERROR) << "swapLast2Axes does not support float32 in current version (C64 only).";
            return nullptr;
        }

        if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64) {
            return GetKernelByName("SwapLast2AxesC64Kernel");
        }

        return nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launchParam, SVector<Tensor>& outTensors) const override
    {
        const Any& specificParam = launchParam.GetParam();
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::SwapLast2Axes), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));

        int64_t dim = static_cast<int64_t>(launchParam.GetInTensor(0).desc.dims.size());
        ASDSIP_CHECK(dim > 1, "invalid dim", return Status::FailStatus(ERROR_ATTR_INVALID_TYPE));

        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;

        int64_t swapLastDimIdx = 2;
        int64_t swapSecondLastDimIdx = 1;

        outTensors[0].desc.dims[dim - swapLastDimIdx] = launchParam.GetInTensor(0)
                                                            .desc.dims[dim - swapSecondLastDimIdx];
        outTensors[0].desc.dims[dim - swapSecondLastDimIdx] = launchParam.GetInTensor(0)
                                                                  .desc.dims[dim - swapLastDimIdx];

        return Status::OkStatus();
    }

    int64_t GetInputNum(const Any& specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::SwapLast2Axes), "OpParam is invalid", return 0);
        return INPUT_DIM;
    }
};
REG_OPERATION(SwapLast2AxesOperation);
} //    namespace AsdSip
