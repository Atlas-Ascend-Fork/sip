/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "complex_mat_dot.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasComplexMatDotPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct ComplexMatDotTensorParam {
    const aclTensor* matx;
    const aclTensor* maty;
    const aclTensor* result;
};

AspbStatus ComplexMatDotDtypeCheck(struct ComplexMatDotTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.matx, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.maty, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.result, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus ComplexMatDotShapeCheck(ComplexMatDotTensorParam param, const int64_t m, const int64_t n)
{
    // m/n 上限收紧（issue #462）：复数 2n 载荷入口（matx/maty/result 为 COMPLEX64，
    // 载荷按实数对计数），对齐 Ccopy/Cswap/Caxpy 的 INT32_MAX/2 口径，防 2m/2n 越过 int32 正域
    ASDSIP_ECHECK(m > 0 && n > 0 && m <= INT32_MAX / 2 && n <= INT32_MAX / 2,
                  "blas asdBlasComplexMatDot get m <= 0 || n <= 0 or m or n > INT32_MAX / 2.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    auto ret = ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    SIP_OP_CHECK_INVALID_SHAPE(param.matx, ret);
    SIP_OP_CHECK_INVALID_SHAPE(param.maty, ret);
    SIP_OP_CHECK_INVALID_SHAPE(param.result, ret);
    SIP_OP_CHECK_NUM_NOT_MATCH(param.matx, m * n, ret);
    SIP_OP_CHECK_NUM_NOT_MATCH(param.maty, m * n, ret);
    SIP_OP_CHECK_NUM_NOT_MATCH(param.result, m * n, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasComplexMatDot(asdBlasHandle handle, const int64_t m, const int64_t n, aclTensor* matx,
                                aclTensor* maty, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas ComplexMatDot get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    struct ComplexMatDotTensorParam parm {
        matx, maty, result
    };
    ASDSIP_CHECK(ComplexMatDotDtypeCheck(parm) == ErrorType::ACL_SUCCESS,
                 "blas asdBlasComplexMatDot dtype check failed.", return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    // shape 校验失败透传内层实际错误码（issue #462，对齐 !372 caxpy 透传口径）：
    // m/n 上界越界返回 INVALID_PARAM，与 NUM_NOT_MATCH 路径的 OP_INPUT_NOT_MATCH 可区分
    auto shapeRet = ComplexMatDotShapeCheck(parm, m, n);
    ASDSIP_CHECK(shapeRet == ErrorType::ACL_SUCCESS, "blas asdBlasComplexMatDot shape check failed.", return shapeRet);

    try {
        AsdSip::BlasComplexMatDotPlan& plan = dynamic_cast<AsdSip::BlasComplexMatDotPlan&>(
            BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "ComplexMatDot plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "ComplexMatDotOperation";
        AsdSip::OpParam::ComplexMatDot param;
        param.m = m;
        param.n = n;
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> complexMatInTensors{matx, maty, plan.augAclTensor};
        SVector<aclTensor*> complexMatOutTensors{result};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, complexMatInTensors, complexMatOutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasComplexMatDot success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the ComplexMat exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "ComplexMatDot Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeComplexMatDotPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeComplexMatDotPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    AsdSip::BlasComplexMatDotPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasComplexMatDotPlan();
        if (!BlasPlanCache::MakePlan(handle, plan)) {
            delete plan;
            ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
                << "blas handle already bound to a plan, repeated initialization is not allowed.";
            return ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    } catch (const std::exception& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make ComplexMatDot Plan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR)
            << "Fail to create blas asdBlasMakeComplexMatDotPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
