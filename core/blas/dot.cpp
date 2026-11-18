/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dot.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasDotPlan.h"

using namespace AsdSip;
using namespace Mki;

constexpr int ELEMENTS_EACH_COMPLEX64 = 2;

namespace AsdSip {

struct DotImplParam {
    int64_t n;
    aclTensor* x;
    int64_t incx;
    aclTensor* y;
    int64_t incy;
    aclTensor* result;
    int64_t isConj;
};

AspbStatus asdBlasDotImpl(asdBlasHandle handle, DotImplParam implParam, int64_t sdot)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Dot get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t dotStorageDimsNum = 0;
    int64_t checkSize = 0;
    auto aclRet = aclGetStorageShape(implParam.x, &storageDims, &dotStorageDimsNum);
    if (aclRet != 0) {
        delete[] storageDims; // 防御性释放: ACL 失败时输出参数状态未知
        ASDSIP_LOG(ERROR) << "asdBlasDotImpl: aclGetStorageShape fail, aclError code is: " << aclRet;
        return ::ACL_ERROR_INTERNAL_ERROR;
    }
    // 先判空再解引用（issue #343）：0 维（标量）张量 dimsNum=0 且 dims 为空指针，
    // 对齐 asdInterpWithCoeff/convolve 的 #245 口径，正常路径零改动
    if (storageDims == nullptr || dotStorageDimsNum < 1) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "asdBlasDotImpl get 0-dim tensor.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    checkSize = sdot == 1 ? *storageDims : *storageDims * ELEMENTS_EACH_COMPLEX64;
    if (checkSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasDotImpl get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    aclRet = aclGetStorageShape(implParam.y, &storageDims, &dotStorageDimsNum);
    if (aclRet != 0) {
        delete[] storageDims;
        ASDSIP_LOG(ERROR) << "asdBlasDotImpl: aclGetStorageShape fail, aclError code is: " << aclRet;
        return ::ACL_ERROR_INTERNAL_ERROR;
    }
    // 先判空再解引用（issue #343）：0 维（标量）张量 dimsNum=0 且 dims 为空指针，
    // 对齐 asdInterpWithCoeff/convolve 的 #245 口径，正常路径零改动
    if (storageDims == nullptr || dotStorageDimsNum < 1) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "asdBlasDotImpl get 0-dim tensor.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    checkSize = sdot == 1 ? *storageDims : *storageDims * ELEMENTS_EACH_COMPLEX64;
    if (checkSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasDotImpl get wrong y tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    ASDSIP_ECHECK(implParam.n > 0, "blas asdBlasDotImpl get wrong input n, please check.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    // 非正步长显式拒绝（issue #260，统一口径）：原静默改写为 1 返回成功，实际计算另一组连续元素，结果静默错误
    ASDSIP_ECHECK(implParam.incx > 0, "blas asdBlasDotImpl get implParam.incx <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    // 非正步长显式拒绝（issue #260，统一口径）：原静默改写为 1 返回成功，实际计算另一组连续元素，结果静默错误
    ASDSIP_ECHECK(implParam.incy > 0, "blas asdBlasDotImpl get implParam.incy <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    Status status;
    OpDesc opDesc;
    opDesc.opName = "DotOperation";
    AsdSip::OpParam::Dot param;
    param.n = implParam.n;
    param.isConj = implParam.isConj;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor*> dotInTensors{implParam.x, implParam.y};
    SVector<aclTensor*> dotOutTensors{implParam.result};

    try {
        AsdSip::BlasDotPlan& plan = dynamic_cast<AsdSip::BlasDotPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Dot plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        status = RunAsdOpsV2(plan.GetStream(), opDesc, dotInTensors, dotOutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasDot success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op dot exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Dot Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasSdot(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                       const int64_t incy, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasSdot: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSdot get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasSdot: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSdot get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasSdot: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSdot get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    // n 上界收紧（issue #424）：原无 n 上限检查，n >= 2^32 时 tiling 侧
    // static_cast<uint32_t>（sdot tiling data n 字段为 uint32_t）回绕为 0 后按错误
    // 长度计算；上限取 INT32_MAX（=UINT32_MAX/2，issue 主口径）：uint32 承载不回绕
    // 且元素偏移不越 int32 正域。实数计数承载不适用 rot #344 的 INT32_MAX/2——
    // 该口径系 2n 载荷 + rot tiling elementCount 为 int32 的特例（dot 族 tiling 为 uint32）
    ASDSIP_ECHECK(n > 0 && n <= INT32_MAX, "blas asdBlasSdot get n <= 0 or n > INT32_MAX.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    DotImplParam param = {n, x, incx, y, incy, result, 0};
    return asdBlasDotImpl(handle, param, 1);
}

AspbStatus asdBlasCdotu(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCdotu: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotu get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasCdotu: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotu get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasCdotu: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotu get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    // n 上界收紧（issue #424）：复数按 ELEMENTS_EACH_COMPLEX64*n=2n 传参，n > INT32_MAX/2
    // 时 2n 越过 uint32 承载（cdot tiling data n 字段为 uint32_t，n=2^32 时 2n=2^33 回绕
    // 为 0）与 int32 元素偏移正域，静默算错，须显式拒绝（对齐 rot #344 的 2n 载荷上限口径）
    ASDSIP_ECHECK(n > 0 && n <= INT32_MAX / 2, "blas asdBlasCdotu get n <= 0 or n > INT32_MAX / 2.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    DotImplParam param = {ELEMENTS_EACH_COMPLEX64 * n, x, incx, y, incy, result, 0};
    return asdBlasDotImpl(handle, param, 0);
}

AspbStatus asdBlasCdotc(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCdotc: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotc get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasCdotc: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotc get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasCdotc: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotc get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    // n 上界收紧（issue #424）：复数按 ELEMENTS_EACH_COMPLEX64*n=2n 传参，n > INT32_MAX/2
    // 时 2n 越过 uint32 承载（cdot tiling data n 字段为 uint32_t，n=2^32 时 2n=2^33 回绕
    // 为 0）与 int32 元素偏移正域，静默算错，须显式拒绝（对齐 rot #344 的 2n 载荷上限口径）
    ASDSIP_ECHECK(n > 0 && n <= INT32_MAX / 2, "blas asdBlasCdotc get n <= 0 or n > INT32_MAX / 2.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    DotImplParam param = {ELEMENTS_EACH_COMPLEX64 * n, x, incx, y, incy, result, 1};
    return asdBlasDotImpl(handle, param, 0);
}

AspbStatus asdBlasMakeDotPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeDotPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasDotPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasDotPlan();
        if (!BlasPlanCache::MakePlan(handle, plan)) {
            delete plan;
            ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
                << "blas handle already bound to a plan, repeated initialization is not allowed.";
            return ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    } catch (const std::exception& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Dot Plan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
