/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "strmv.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasStrmvPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct StrmvTensorParam {
    const aclTensor* A;
    const aclTensor* x;
};

AspbStatus StrmvDtypeCheck(struct StrmvTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_FLOAT, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus StrmvShapeCheck(struct StrmvTensorParam parm, const int64_t n)
{
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasStrmv get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    // 维度数校验：0 维张量（dims 为空）会使 tiling 层 dims.at(0) 抛 std::out_of_range（issue #139）
    SIP_OP_CHECK_INVALID_SHAPE(parm.A, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_INVALID_SHAPE(parm.x, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.A, n * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.x, n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

// 入口参数护栏与 tiling 参数映射（issue #125/#423/#422）：非法 incx/lda/trans/diag
// 枚举显式拒绝而非静默按默认值计算（原 else 映射把 OP_C 与越界枚举静默按转置、
// diag 非法值按 NON_UNIT 计算，接口假成功），合法值映射 tiling 参数
AspbStatus StrmvCheckAndMapParam(asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag,
                                 const int64_t n, const int64_t lda, const int64_t incx, AsdSip::OpParam::Strmv& param)
{
    ASDSIP_ECHECK(incx == 1, "blas asdBlasStrmv only supports incx == 1 currently.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(lda == n, "blas asdBlasStrmv only supports lda == n currently.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(
        uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER || uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER,
        "blas asdBlasStrmv get invalid uplo (only LOWER/UPPER supported).", ErrorType::ACL_ERROR_INVALID_PARAM);
    // 非法 trans/diag 枚举显式拒绝（issue #423，对齐 ctrmv/strmm 的 #345 加固口径）：strmv 为实数算子，OP_C 无意义
    ASDSIP_ECHECK(trans == asdBlasOperation_t::ASDBLAS_OP_N || trans == asdBlasOperation_t::ASDBLAS_OP_T,
                  "blas asdBlasStrmv get invalid trans enum.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(diag == asdBlasDiagType_t::ASDBLAS_DIAG_UNIT || diag == asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT,
                  "blas asdBlasStrmv get invalid diag enum.", ErrorType::ACL_ERROR_INVALID_PARAM);
    param.trans = (trans == asdBlasOperation_t::ASDBLAS_OP_N) ? 0 : 1;
    param.uplo = (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER) ? 1 : 0;
    param.diag = (diag == asdBlasDiagType_t::ASDBLAS_DIAG_UNIT) ? 1 : 0;
    param.incx = 1;
    param.lda = n;
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasStrmv(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag,
                        const int64_t n, aclTensor* A, const int64_t lda, aclTensor* x, const int64_t incx)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Strmv get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct StrmvTensorParam parm {
        A, x
    };
    ASDSIP_CHECK(StrmvDtypeCheck(parm) == ErrorType::ACL_SUCCESS, "blas asdBlasStrmv dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    auto ret = StrmvShapeCheck(parm, n);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasStrmv shape check failed.", return ret);

    AsdSip::OpParam::Strmv param = {0, 0, 0, 0, 0};
    AspbStatus pstat = StrmvCheckAndMapParam(uplo, trans, diag, n, lda, incx, param);
    ASDSIP_CHECK(pstat == ErrorType::ACL_SUCCESS, "blas asdBlasStrmv param check failed.", return pstat);

    try {
        AsdSip::BlasStrmvPlan& plan = dynamic_cast<AsdSip::BlasStrmvPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Strmv plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        // plan 一致性校验（issue #422，对齐 smatinv_batched 的 plan.GetN() 校验口径）：
        // 掩码张量按 MakePlan 时的 uplo 生成、workspace 按该 n 分配，调用侧 uplo/n 与
        // plan 不一致时三角选取消隐错乱或越界写 workspace，须显式拦截
        ASDSIP_ECHECK(plan.GetUplo() == uplo && plan.GetN() == n,
                      "blas asdBlasStrmv uplo/n mismatch with plan (plan was made with different uplo/n, "
                      "re-make plan first).",
                      ErrorType::ACL_ERROR_INVALID_PARAM);

        OpDesc opDesc;
        opDesc.opName = "StrmvOperation";
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{A, x, plan.GetMaskAclTensor()};
        SVector<aclTensor*> outTensors{x};

        Status status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasStrmv success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op strmv exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Strmv Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeStrmvPlan(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, int64_t n)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeStrmvPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasStrmvPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasStrmvPlan(uplo, trans, n);
        if (!BlasPlanCache::MakePlan(handle, plan)) {
            delete plan;
            ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
                << "blas handle already bound to a plan, repeated initialization is not allowed.";
            return ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    } catch (const std::exception& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Strmv Plan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeStrmvPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
