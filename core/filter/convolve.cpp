/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/assert.h"
#include "log/log.h"
#include "filter_api.h"
#include "utils/ops_base.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/common_types.h"
#include "convolve.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {

constexpr uint64_t DIMS_TWO = 2;
constexpr uint64_t BASE_COL_BLOCK = 128;

AsdSip::AspbStatus asdConvolve(const aclTensor* signal, const aclTensor* kernel, aclTensor* output,
                               asdConvolveMode_t mode, void* stream, void* workspace)
{
    // mode 值域校验（issue #246）：头文件定义 FULL/VALID/SAME 三种模式，文档声明
    // 当前仅支持 SAME；原 (void)mode 丢弃使 FULL/VALID 静默按 SAME 计算并返回成功
    if (mode != asdConvolveMode_t::ASD_CONVOLVE_SAME) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
            << "asdConvolve only supports ASD_CONVOLVE_SAME in current version.";
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(
        aclGetStorageShape(static_cast<const aclTensor*>(signal), &storageDims, &storageDimsNum),
        "asdConvolve: aclGetStorageShape");
    // 先判空再解引用（issue #245）：0 维张量 dims 为空指针
    if (storageDims == nullptr || storageDimsNum < 1 || *storageDims <= 0 || storageDimsNum != DIMS_TWO) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "asdConvolve get wrong inputs.";
        return AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    int64_t batchCount = storageDims[0];
    int64_t signalLen = storageDims[1];

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(
        aclGetStorageShape(static_cast<const aclTensor*>(kernel), &storageDims, &storageDimsNum),
        "asdConvolve: aclGetStorageShape");
    // 先判空再解引用（issue #245，同 signal 侧口径）
    if (storageDims == nullptr || storageDimsNum < 1 || *storageDims <= 0 || storageDimsNum > DIMS_TWO) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "asdConvolve get wrong inputs.";
        return AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    int64_t kernelLen = storageDims[0];

    delete[] storageDims;
    storageDims = nullptr;

    op::DataType dataType = op::DataType::DT_UNDEFINED;
    dataType = signal->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_COMPLEX64 || dataType == op::DataType::DT_COMPLEX32,
                  "asdConvolve get wrong signal tensor dtype.", ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    dataType = op::DataType::DT_UNDEFINED;
    dataType = kernel->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_FLOAT || dataType == op::DataType::DT_FLOAT16,
                  "asdConvolve get wrong kernel tensor dtype.", ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    dataType = op::DataType::DT_UNDEFINED;
    dataType = output->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_COMPLEX64 || dataType == op::DataType::DT_COMPLEX32,
                  "asdConvolve get wrong output tensor dtype.", ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    // signal 与 output 必须同 dtype（issue #304）：param.dataType 仅由 output 决定，
    // 混用 complex64/complex32 时 kernel 以错误元素宽度解读 signal 缓冲，输出整体
    // 错误且无报错
    ASDSIP_ECHECK(signal->GetDataType() == output->GetDataType(),
                  "asdConvolve signal and output tensor dtype mismatch.", ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    OpDesc opDesc;
    opDesc.opName = "ConvolveOperation";
    AsdSip::OpParam::Convolve param;
    param.batchCount = batchCount;
    param.signalLen = signalLen;
    param.kernelLen = kernelLen;
    param.dataType = (dataType == op::DataType::DT_COMPLEX64) ? 0 : 1;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor*> inTensors{const_cast<aclTensor*>(static_cast<const aclTensor*>(signal)),
                                  const_cast<aclTensor*>(static_cast<const aclTensor*>(kernel))};
    SVector<aclTensor*> outTensors{const_cast<aclTensor*>(static_cast<const aclTensor*>(output))};

    Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, reinterpret_cast<uint8_t*>(workspace));
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_LOG(INFO) << "Execute asdConvolve success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdConvolveGetWorkspaceSize(int64_t signalLen, int64_t kernelLen, size_t& size)
{
    ASDSIP_ECHECK(signalLen > 0 && kernelLen > 0, "asdConvolveGetWorkspaceSize get invalid params",
                  AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    if (kernelLen > static_cast<int64_t>(SIZE_MAX / DIMS_TWO / DIMS_TWO)) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
            << "asdConvolveGetWorkspaceSize get too large kernelLen: " << kernelLen;
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    size_t kernelSize = static_cast<size_t>(kernelLen);
    size_t workspaceColNum = BASE_COL_BLOCK + kernelSize * DIMS_TWO - DIMS_TWO;
    size_t workspaceRowNum = workspaceColNum + kernelSize * DIMS_TWO - DIMS_TWO;
    // 按 2 * sizeof(float)（complex64 的 8B/元素）为上界计算：tiling 侧对 complex64 按 8B/元素
    // 分配 scratch、内核按 2×float 间距寻址 workspace，原 4B/元素对 complex64 场景会导致
    // 设备侧越界写（issue #280）；complex32 用户多分配一倍属安全冗余
    if (workspaceRowNum > SIZE_MAX / workspaceColNum ||
        workspaceColNum * workspaceRowNum > SIZE_MAX / (DIMS_TWO * sizeof(float))) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
            << "asdConvolveGetWorkspaceSize overflow, kernelLen: " << kernelLen;
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    size = workspaceColNum * workspaceRowNum * DIMS_TWO * sizeof(float);

    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
