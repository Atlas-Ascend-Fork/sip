/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cgerc.h"
#include <complex>
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCgercPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct CgercTensorParam {
    const aclTensor* x;
    const aclTensor* y;
    const aclTensor* A;
};

AspbStatus CgercDtypeCheck(struct CgercTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.y, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CgercShapeCheck(struct CgercTensorParam parm, const int64_t m, const int64_t n)
{
    // m/n 上限收紧（issue #462）：复数 2n 载荷入口（x/y 为 m/n 个 COMPLEX64，2m/2n 个
    // 实数载荷），对齐 Ccopy/Cswap/Caxpy 的 INT32_MAX/2 口径，防 2m/2n 越过 int32 正域
    ASDSIP_ECHECK(m > 0 && n > 0 && m <= INT32_MAX / 2 && n <= INT32_MAX / 2,
                  "blas Cgerc get m <= 0 || n <= 0 or m or n > INT32_MAX / 2.", ErrorType::ACL_ERROR_INVALID_PARAM);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.A, m * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.x, m, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.y, n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCgerc(asdBlasHandle handle, const int64_t m, const int64_t n, const std::complex<float>& alpha,
                        aclTensor* x, const int64_t incx, aclTensor* y, const int64_t incy, aclTensor* A,
                        const int64_t lda)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Cgerc get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct CgercTensorParam parm {
        x, y, A
    };
    ASDSIP_CHECK(CgercDtypeCheck(parm) == ErrorType::ACL_SUCCESS, "blas asdBlasCgerc dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    // shape 校验失败透传内层实际错误码（issue #462，对齐 !372 caxpy 透传口径）：
    // m/n 上界越界返回 INVALID_PARAM，与 NUM_NOT_MATCH 路径的 OP_INPUT_NOT_MATCH 可区分
    auto ret = CgercShapeCheck(parm, m, n);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasCgerc shape check failed.", return ret);

    // 非正步长显式拒绝（issue #243）：负值经 static_cast<uint32_t> 回绕为超大
    // 步长传入内核产生非法寻址；非 1 正步长内核同样不支持，一并对齐 cgemv 的
    // !=1 显式拒绝口径
    ASDSIP_ECHECK(incx > 0 && incx == 1, "blas asdBlasCgerc get incx <= 0 or != 1 (strided access not supported).",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(incy > 0 && incy == 1, "blas asdBlasCgerc get incy <= 0 or != 1 (strided access not supported).",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    // 非法 leading dimension 显式拒绝（issue #385，对齐本文件 incx/incy 显式拒绝口径）：
    // 原 lda <= 0 仅打日志后继续执行，非法输入得到"成功"返回，结果静默错误
    // 非约定 leading dimension 显式拒绝（issue #484，对齐 cgemv #461 的 lda==m 口径）：
    // lda 未进 OpParam、tiling/kernel 恒按紧凑布局计算，lda != m 时与 A 元素数 = m*n
    // 契约不可同时满足（lda < m 列区间重叠静默算错、lda > m 越过元素存储），
    // 仅放行 lda == m（docs Cgerc.md lda 当前约束为 m）
    ASDSIP_ECHECK(lda > 0 && lda == m, "blas asdBlasCgerc get lda <= 0 or lda != m.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    try {
        AsdSip::BlasCgercPlan& plan = dynamic_cast<AsdSip::BlasCgercPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Cgerc plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        OpDesc opDesc;
        opDesc.opName = "CgercOperation";
        AsdSip::OpParam::Cgerc param;
        param.m = m;
        param.n = n;
        param.incx = static_cast<uint32_t>(incx);
        param.incy = static_cast<uint32_t>(incy);
        param.alphaReal = alpha.real();
        param.alphaImag = alpha.imag();
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{x, y, plan.gatherAclOffset};
        SVector<aclTensor*> outTensors{A};
        Status status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCgerc success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op cgerc exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Cgerc Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeCgercPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeCgercPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasCgercPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCgercPlan();
        if (!BlasPlanCache::MakePlan(handle, plan)) {
            delete plan;
            ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
                << "blas handle already bound to a plan, repeated initialization is not allowed.";
            return ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    } catch (const std::exception& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Cgerc Plan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeCgercPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
