/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "smatinv_batched.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasSmatinvBatchedPlan.h"
#include "blasplan/include/blasplan/BlasCgetriBatchedPlan.h"
#include "sgetri_batched.h"

using namespace AsdSip;
using namespace Mki;

static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;

static constexpr int32_t MATRIX_SHAPE_LIMIT = 32;
static constexpr int32_t MAX_MATRIX_SHAPE = 256;
static constexpr int32_t MAX_MATRIX_BATCH = 3000;

static constexpr int64_t DTYPE_FLOAT32 = 0;
static constexpr int64_t DTYPE_FLOAT16 = 1;

namespace AsdSip {

struct SmatinvBatchedImplParam {
    int64_t dtype;
    int64_t n;
    int64_t batchSize;
    aclTensor* A;
    aclTensor* Ainv;
};

AspbStatus asdBlasSmatinvBatchedShapeCheck(SmatinvBatchedImplParam implParam)
{
    int64_t* AstorageDims = nullptr;
    uint64_t AstorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.A, &AstorageDims, &AstorageDimsNum),
                                 "asdBlasSmatinvBatched: aclGetStorageShape");
    if (AstorageDimsNum != DIM_3 || AstorageDims[DIM_0] != implParam.batchSize || AstorageDims[DIM_1] != implParam.n ||
        AstorageDims[DIM_2] != implParam.n) {
        delete[] AstorageDims;
        AstorageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH)
            << "blas asdBlasSmatinvBatched get wrong A tensor dim num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (AstorageDims != nullptr) {
        delete[] AstorageDims;
        AstorageDims = nullptr;
    }

    int64_t* AinvStorageDims = nullptr;
    uint64_t AinvStorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.Ainv, &AinvStorageDims, &AinvStorageDimsNum),
                                 "asdBlasSmatinvBatched: aclGetStorageShape");
    if (AinvStorageDimsNum != DIM_3 || AinvStorageDims[DIM_0] != implParam.batchSize ||
        AinvStorageDims[DIM_1] != implParam.n || AinvStorageDims[DIM_2] != implParam.n) {
        delete[] AinvStorageDims;
        AinvStorageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH)
            << "blas asdBlasSmatinvBatched get wrong Ainv tensor dim num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (AinvStorageDims != nullptr) {
        delete[] AinvStorageDims;
        AinvStorageDims = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

// n >= 32: 实数走 SgetriBatched (真实数 Cgetri 变体: 入口直拷不拆实虚,
// imag plane 全零填充, 出口只拷 real plane; LU/trsm/getri 流水复用)
static constexpr int64_t DTYPE_REAL32 = 2;

AspbStatus runSgetriBatchedOps(asdBlasHandle handle, SmatinvBatchedImplParam implParam)
{
    try {
        AsdSip::BlasCgetriBatchedPlan& plan = dynamic_cast<AsdSip::BlasCgetriBatchedPlan&>(
            BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "SgetriBatched plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        // plan 的 W/eye/A_work 等 tensor 与 workspace 均按 MakePlan 时的
        // n/batchSize 分配, 执行参数与 plan 不一致会导致 kernel 越界, 须显式拦截
        ASDSIP_ECHECK(plan.GetN() == implParam.n && plan.GetBatchSize() == implParam.batchSize,
                      "blas asdBlasSmatinvBatched n/batchSize mismatch with plan "
                      "(plan was made with different n/batchSize, re-make plan first).",
                      ErrorType::ACL_ERROR_INVALID_PARAM);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "SgetriBatchedOperation";
        AsdSip::OpParam::SgetriBatched param;

        param = {DTYPE_REAL32, implParam.n, implParam.batchSize};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{implParam.A,           plan.wBatchAcl,        plan.gather1OffsetAcl,
                                      plan.gather2OffsetAcl, plan.gather3OffsetAcl, plan.eyeBatchMatAcl,
                                      plan.aWorkAcl,         plan.workGmAcl};
        SVector<aclTensor*> outTensors{implParam.Ainv};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasSmatinvBatched success when n >= 32 (SgetriBatched).";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "SgetriBatched Error (n >= 32): " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus makeSgetriBatchedPlan(asdBlasHandle handle, const int64_t n, const int64_t batchSize)
{
    // 复用 BlasCgetriBatchedPlan (dtype 只影响 workspace: real 走 else 分支最小 rsv)
    AsdSip::BlasCgetriBatchedPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCgetriBatchedPlan(n, batchSize, DTYPE_REAL32);
        if (!BlasPlanCache::MakePlan(handle, plan)) {
            delete plan;
            ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
                << "blas handle already bound to a plan, repeated initialization is not allowed.";
            return ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    } catch (const std::exception& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make SgetriBatched Plan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas SgetriBatched mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}

AspbStatus runSmatinvBatchedOps(asdBlasHandle handle, SmatinvBatchedImplParam implParam)
{
    try {
        AsdSip::BlasSmatinvBatchedPlan& plan = dynamic_cast<AsdSip::BlasSmatinvBatchedPlan&>(
            BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "SmatinvBatched plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        // GJ 路径 plan tensor (单位阵) 仅按 n 分配, batchSize 由执行时张量形状
        // 决定, 故只校验 n 与 plan 一致
        ASDSIP_ECHECK(plan.GetN() == implParam.n,
                      "blas asdBlasSmatinvBatched n mismatch with plan "
                      "(plan was made with different n, re-make plan first).",
                      ErrorType::ACL_ERROR_INVALID_PARAM);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "SmatinvBatchedOperation";
        AsdSip::OpParam::SmatinvBatched param;

        param = {implParam.dtype, implParam.n, implParam.batchSize};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{implParam.A, plan.uniMatAclReal};
        SVector<aclTensor*> outTensors{implParam.Ainv};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasSmatinvBatched success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "SmatinvBatched Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasSmatinvBatchedImpl(asdBlasHandle handle, SmatinvBatchedImplParam implParam)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas SmatinvBatched get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    AsdSip::AspbStatus shapeCheckStatus = asdBlasSmatinvBatchedShapeCheck(implParam);
    if (shapeCheckStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        return shapeCheckStatus;
    }

    // n < 32: Gauss-Jordan 消元 kernel (小矩阵最优)
    // 32 <= n <= 256: SgetriBatched 真实数 LU 求逆 kernel (入口直拷不拆实虚)
    if (implParam.n < MATRIX_SHAPE_LIMIT) {
        return runSmatinvBatchedOps(handle, implParam);
    }
    return runSgetriBatchedOps(handle, implParam);
}

AspbStatus asdBlasSmatinvBatched(asdBlasHandle handle, const int64_t n, aclTensor* A, const int64_t lda,
                                 aclTensor* Ainv, const int64_t lda_inv, aclTensor* info, int64_t batchSize)
{
    (void)info;
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(A, &dataType), "asdBlasSmatinvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSmatinvBatched get wrong A tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(Ainv, &dataType), "asdBlasSmatinvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSmatinvBatched get wrong Ainv tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    ASDSIP_ECHECK(lda > 0 && lda_inv > 0, "blas asdBlasSmatinvBatched get lda <= 0 or lda_inv <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(lda == n && lda_inv == n, "blas asdBlasSmatinvBatched only supports lda == n and lda_inv == n.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_ECHECK(n > 0 && batchSize > 0, "blas asdBlasSmatinvBatched get n <= 0 || batchSize <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "blas asdBlasSmatinvBatched requires n <= 256 and batchSize <= 3000.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    return asdBlasSmatinvBatchedImpl(handle, {DTYPE_FLOAT32, n, batchSize, A, Ainv});
}

AspbStatus asdBlasHSmatinvBatched(asdBlasHandle handle, const int64_t n, aclTensor* A, const int64_t lda,
                                  aclTensor* Ainv, const int64_t lda_inv, aclTensor* info, int64_t batchSize)
{
    (void)info;
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(A, &dataType), "asdBlasHSmatinvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT16, "blas asdBlasHSmatinvBatched get wrong A tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(Ainv, &dataType), "asdBlasHSmatinvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT16, "blas asdBlasHSmatinvBatched get wrong Ainv tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    ASDSIP_ECHECK(lda > 0 && lda_inv > 0, "blas asdBlasHSmatinvBatched get lda <= 0 or lda_inv <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(lda == n && lda_inv == n, "blas asdBlasHSmatinvBatched only supports lda == n and lda_inv == n.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_ECHECK(n > 0 && batchSize > 0, "blas asdBlasHSmatinvBatched get n <= 0 || batchSize <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    // fp16 仅小矩阵消元 kernel; n >= 32 的 Sgetri 为 fp32 kernel, 由上层转 fp32 走 F32 入口
    ASDSIP_ECHECK(n < MATRIX_SHAPE_LIMIT, "blas asdBlasHSmatinvBatched only supports n < 32 (fp16).",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "blas asdBlasHSmatinvBatched requires n <= 256 and batchSize <= 3000.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    return asdBlasSmatinvBatchedImpl(handle, {DTYPE_FLOAT16, n, batchSize, A, Ainv});
}

AspbStatus asdBlasMakeSmatinvBatchedPlan(asdBlasHandle handle, const int64_t n, const int64_t batchSize)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas SmatinvBatched Make Plan failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次（issue #129 同款防护）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(n > 0, "blas asdBlasMakeSmatinvBatchedPlan get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(batchSize > 0, "blas asdBlasMakeSmatinvBatchedPlan get batchSize <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(n <= MAX_MATRIX_SHAPE, "blas asdBlasMakeSmatinvBatchedPlan requires n <= 256.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    // n >= 32: SgetriBatched plan (复用 BlasCgetriBatchedPlan, dtype=real32)
    if (n >= MATRIX_SHAPE_LIMIT) {
        return makeSgetriBatchedPlan(handle, n, batchSize);
    }

    AsdSip::BlasSmatinvBatchedPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasSmatinvBatchedPlan(n);
        if (!BlasPlanCache::MakePlan(handle, plan)) {
            delete plan;
            ASDSIP_ELOG(ErrorType::ACL_ERROR_INVALID_PARAM)
                << "blas handle already bound to a plan, repeated initialization is not allowed.";
            return ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    } catch (const std::exception& e) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make SmatinvBatched Plan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas SmatinvBatched identity tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
