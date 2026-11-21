/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ctrmv.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCtrmvPlan.h"

using namespace AsdSip;
using namespace Mki;

constexpr int TWO = 2;

namespace AsdSip {
struct CtrmvTensorParam {
    const aclTensor* A;
    const aclTensor* x;
};

AspbStatus CtrmvDtypeCheck(struct CtrmvTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CtrmvShapeCheck(CtrmvTensorParam param, const int64_t n, const int64_t incx, const int64_t lda)
{
    // n 上限收紧（issue #462）：复数 2n 载荷入口（x 为 n 个 COMPLEX64，2n 个实数载荷），
    // 对齐 Ccopy/Cswap/Caxpy 的 INT32_MAX/2 口径，防 n > INT32_MAX/2 时 2n 越过 int32 正域
    ASDSIP_ECHECK(n > 0 && n <= INT32_MAX / 2, "blas asdBlasCtrmv get n <= 0 or n > INT32_MAX / 2.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(incx > 0, "blas asdBlasCtrmv get incx < 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(lda > 0, "blas asdBlasCtrmv get lda <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(lda == n, "blas asdBlasCtrmv only supports lda == n.", ErrorType::ACL_ERROR_INVALID_PARAM);
    auto ret = ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    SIP_OP_CHECK_NUM_NOT_MATCH(param.A, n * n, ret);
    SIP_OP_CHECK_NUM_NOT_MATCH(param.x, n, ret);
    return ErrorType::ACL_SUCCESS;
}

// 入口枚举校验与 tiling 参数映射（issue #345/#422）：非法枚举显式拒绝而非静默映射
// （原 transLocal=-1 透传 tiling，内核对非 1/2 回落 OP_N），合法值映射 tiling 参数
AspbStatus CtrmvCheckAndMapParam(asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag,
                                 const int64_t n, const int64_t lda, const int64_t incx, AsdSip::OpParam::Ctrmv& param)
{
    int64_t transLocal = -1;
    if (trans == asdBlasOperation_t::ASDBLAS_OP_N) {
        transLocal = 0;
    } else if (trans == asdBlasOperation_t::ASDBLAS_OP_T) {
        transLocal = 1;
    } else if (trans == asdBlasOperation_t::ASDBLAS_OP_C) {
        transLocal = TWO;
    } else {
        // 非法枚举显式拒绝（issue #345，对齐 cgemm 的 trans 校验口径）：原 transLocal=-1
        // 透传 tiling，内核对非 1/2 回落 OP_N，静默按非转置计算并返回成功
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM) << "blas asdBlasCtrmv get invalid trans enum.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // 非法枚举显式拒绝（issue #345）：原三目映射把非法 uplo/diag 静默映射为 LOWER/UNIT
    ASDSIP_ECHECK(
        uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER || uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER,
        "blas asdBlasCtrmv get invalid uplo enum.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(diag == asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT || diag == asdBlasDiagType_t::ASDBLAS_DIAG_UNIT,
                  "blas asdBlasCtrmv get invalid diag enum.", ErrorType::ACL_ERROR_INVALID_PARAM);

    param.uplo = (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER) ? 0 : 1;
    param.trans = transLocal;
    param.diag = (diag == asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT) ? 0 : 1;
    param.n = n;
    param.lda = lda;
    param.incx = incx;
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCtrmv(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag,
                        const int64_t n, aclTensor* A, const int64_t lda, aclTensor* x, const int64_t incx)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Ctrmv get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct CtrmvTensorParam parm {
        A, x
    };
    ASDSIP_CHECK(CtrmvDtypeCheck(parm) == ErrorType::ACL_SUCCESS, "blas asdBlasCtrmv dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    auto ret = CtrmvShapeCheck(parm, n, incx, lda);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasCtrmv shape check failed.", return ret);

    AsdSip::OpParam::Ctrmv param;
    AspbStatus pstat = CtrmvCheckAndMapParam(uplo, trans, diag, n, lda, incx, param);
    ASDSIP_CHECK(pstat == ErrorType::ACL_SUCCESS, "blas asdBlasCtrmv param check failed.", return pstat);

    Status status;
    OpDesc opDesc;
    opDesc.opName = "CtrmvOperation";
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    try {
        AsdSip::BlasCtrmvPlan& plan = dynamic_cast<AsdSip::BlasCtrmvPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Ctrmv plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        // plan 一致性校验（issue #422，对齐 smatinv_batched 的 plan.GetN() 校验口径）：
        // uplo 掩码张量与 workspace 均按 MakePlan 时的 uplo/n 生成，调用侧参数与 plan
        // 不一致时三角选取消隐错乱或越界写 workspace，须显式拦截
        ASDSIP_ECHECK(plan.GetUplo() == uplo && plan.GetN() == n,
                      "blas asdBlasCtrmv uplo/n mismatch with plan (plan was made with different uplo/n, "
                      "re-make plan first).",
                      ErrorType::ACL_ERROR_INVALID_PARAM);

        SVector<aclTensor*> inTensors{A, x, plan.GetUploAclTensor()};
        SVector<aclTensor*> outTensors;
        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCtrmv success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the BlasCtrmv exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Ctrmv Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeCtrmvPlan(asdBlasHandle handle, asdBlasFillMode_t uplo, int64_t n)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeCtrmvPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    // n 上限收紧（issue #462）：与 CtrmvShapeCheck 同口径（复数 2n 载荷，INT32_MAX/2）
    ASDSIP_ECHECK(n > 0 && n <= INT32_MAX / 2, "blas asdBlasCtrmv get n <= 0 or n > INT32_MAX / 2.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    AsdSip::BlasCtrmvPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCtrmvPlan(uplo, n);
        if (!BlasPlanCache::MakePlan(handle, plan)) {
            delete plan;
            ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
                << "blas handle already bound to a plan, repeated initialization is not allowed.";
            return ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    } catch (const std::exception& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Ctrmv Plan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeCtrmvPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
