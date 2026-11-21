/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "domain/rs_api.h"
#include "utils/ops_base.h"
#include "utils/op_desc.h"
#include "utils/aspb_status.h"
#include "utils/assert.h"
#include "log/log.h"
#include "utils/common_check.h"
#include "mki/tensor.h"
#include "mki/utils/rt/rt.h"
#include "mki/utils/assert/assert.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "interpolation.h"

using namespace AsdSip;
using namespace Mki;

// COMPUTE_PER_NUM : 16
// ALPHA : 4
// usedCubeCoreNum ：20
// WORKSPACE_SIZE : (ALPHA * COMPUTE_PER_NUM * 2) * (COMPUTE_PER_NUM * 2) + 16 + (ALPHA * COMPUTE_PER_NUM + 8) * 2;
// WORKSPACE_SIZE * usedCubeCoreNum * sizeof(float) * 2;
constexpr uint32_t INTERPOLATION_WORKSPACE_SIZE = 680960; // 4256 * 20 * 4 * 2
static constexpr uint32_t DIM_2 = 2;
// 内核固定块长（tiling COMPUTE_PER_NUM，issue #457）
constexpr int32_t INTERP_BLOCK_NUM = 16;
// 内核 sincTab 打包布局绑定的参数组合（issue #458）
constexpr int32_t SUPPORTED_TAB_INTERP_NUM = 16;
constexpr int32_t SUPPORTED_TAB_QUANT_NUM = 32;

namespace AsdSip {

struct InterpolationTensorParam {
    const aclTensor* inputTensor;
    const aclTensor* sincTab;
    const aclTensor* posFloor;
    const aclTensor* posToTabIndex;
    aclTensor* outputTensor;
};

AspbStatus InterpolationDtypeCheck(InterpolationTensorParam param)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(param.inputTensor, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(param.sincTab, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(param.posFloor, aclDataType::ACL_INT32, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(param.posToTabIndex, aclDataType::ACL_INT16, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(param.outputTensor, aclDataType::ACL_COMPLEX64, ret);

    return ErrorType::ACL_SUCCESS;
}

AspbStatus InterpolationShapeCheck(InterpolationTensorParam param)
{
    auto ret = ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    SIP_OP_CHECK_INVALID_SHAPE(param.inputTensor, ret);
    SIP_OP_CHECK_INVALID_SHAPE(param.sincTab, ret);
    SIP_OP_CHECK_INVALID_SHAPE(param.posFloor, ret);
    SIP_OP_CHECK_INVALID_SHAPE(param.posToTabIndex, ret);
    SIP_OP_CHECK_INVALID_SHAPE(param.outputTensor, ACL_ERROR_OP_OUTPUT_NOT_MATCH);

    return ErrorType::ACL_SUCCESS;
}

// tab 布局与 interpLength 约束校验（issue #457/#458）：内核 sincTab 打包布局按
// interpNum=16/quantNum=32 固定排布（TAB_OFFSET 与系数行步长为内核编译期常量），
// 其余参数组合会按固定布局越界读取/错位取行；内核按固定块长 INTERP_BLOCK_NUM 整除
// 分块，非 16 倍数的 interpLength 尾块不进计算循环，输出残留 workspace 旧值
AspbStatus InterpolationParamCheck(int interpNum, int quantNum, int interpLength)
{
    ASDSIP_ECHECK(interpNum == SUPPORTED_TAB_INTERP_NUM && quantNum == SUPPORTED_TAB_QUANT_NUM,
                  "rsInterpolationBySinc only supports interpNum=16 and quantNum=32 in current version.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(interpLength > 0 && interpLength % INTERP_BLOCK_NUM == 0,
                  "rsInterpolationBySinc get interpLength <= 0 or not a multiple of 16.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus rsInterpolationBySinc(const aclTensor* inputTensor, const aclTensor* sincTab, const aclTensor* posFloor,
                                 const aclTensor* posToTabIndex, aclTensor* outputTensor, int interpNum, int quantNum,
                                 int interpLength, void* stream, void* workSpace)
{
    struct InterpolationTensorParam parm = {inputTensor, sincTab, posFloor, posToTabIndex, outputTensor};
    auto ret = InterpolationDtypeCheck(parm);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }

    ret = InterpolationShapeCheck(parm);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }

    ret = InterpolationParamCheck(interpNum, quantNum, interpLength);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }

    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(inputTensor, &storageDims, &storageDimsNum),
                                 "rsInterpolationBySinc: aclGetStorageShape");

    if (storageDimsNum != DIM_2) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "Interpolation input tensor must be 2-dimensional.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    OpDesc opDesc;
    opDesc.opName = "InterpolationOperation";
    AsdSip::OpParam::Interpolation param;
    param.tabInterpNum = interpNum;
    param.tabQuantNum = quantNum;
    param.batch = storageDims[0];
    param.signalLength = storageDims[1];
    param.interpLength = interpLength;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    SVector<aclTensor*> inTensors{const_cast<aclTensor*>(inputTensor), const_cast<aclTensor*>(sincTab),
                                  const_cast<aclTensor*>(posFloor), const_cast<aclTensor*>(posToTabIndex)};

    SVector<aclTensor*> outTensors{outputTensor};

    Mki::Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, (uint8_t*)workSpace);
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_LOG(INFO) << "Execute interpolation success.";
    return ErrorType::ACL_SUCCESS;
}

AspbStatus rsInterpolationBySincGetWorkspaceSize(size_t& workspaceSize)
{
    workspaceSize = INTERPOLATION_WORKSPACE_SIZE;
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
