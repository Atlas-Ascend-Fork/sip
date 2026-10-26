/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_BLAS_API_H
#define ASDSIP_BLAS_API_H

#include <complex>
#include "aclnn/opdev/fp16_t.h"
#include "acl/acl.h"
#include "utils/aspb_status.h"
#include "utils/mem_base.h"

namespace AsdSip {
using asdBlasHandle = void*;

enum class asdBlasStatus { BLAS_SUCCESS = 0, BLAS_FAILED = 1 };

enum class asdBlasSideMode_t { ASDBLAS_SIDE_LEFT = 0, ASDBLAS_SIDE_RIGHT };

enum class asdBlasOperation_t { ASDBLAS_OP_N = 0, ASDBLAS_OP_T, ASDBLAS_OP_C };

enum class asdBlasFillMode_t { ASDBLAS_FILL_MODE_LOWER = 0, ASDBLAS_FILL_MODE_UPPER, ASDBLAS_FILL_MODE_FULL };

enum class asdBlasDiagType_t { ASDBLAS_DIAG_NON_UNIT = 0, ASDBLAS_DIAG_UNIT };

// Creates only an opaque handle, and only allocates space basic data structures for it.
// This function does not initialize handle.
AspbStatus asdBlasCreate(asdBlasHandle& handle);

// Binds a stream to handle.
AspbStatus asdBlasSetStream(asdBlasHandle handle, void* stream);

// Initializes handle with plan.
// Any given handle can only be initialized once.
AspbStatus asdBlasMakeSsyr2Plan(asdBlasHandle handle);

AspbStatus asdBlasMakeSsyrPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeCgemmPlan(asdBlasHandle handle, asdBlasOperation_t transa, asdBlasOperation_t transb, int64_t m,
                                int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc);

AspbStatus asdBlasMakeStrmvPlan(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, int64_t n);

AspbStatus asdBlasMakeCgemvPlan(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m, const int64_t n,
                                aclTensor* y, const int64_t incy);

AspbStatus asdBlasMakeCtrmvPlan(asdBlasHandle handle, asdBlasFillMode_t uplo, int64_t n);

AspbStatus asdBlasMakeAsumPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeDotPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeNrm2Plan(asdBlasHandle handle);

AspbStatus asdBlasMakeCopyPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeCalPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeColwiseMulPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeComplexMatDotPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeIamaxPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeCaxpyPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeStrmmPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeCgercPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeRotPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeSwapPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeHCgemvBatchedPlan(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m);

AspbStatus asdBlasMakeCgemvBatchedPlan(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m);

AspbStatus asdBlasMakeHCgemmBatchedPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeCgemmBatchedPlan(asdBlasHandle handle);

AspbStatus asdBlasMakeHCmatinvBatchedPlan(asdBlasHandle handle, const int64_t n, const int64_t batchSize);

AspbStatus asdBlasMakeCmatinvBatchedPlan(asdBlasHandle handle, const int64_t n, const int64_t batchSize);

// batched real-matrix inversion (float32 / float16), n < 32, batchSize <= 3000
AspbStatus asdBlasMakeSmatinvBatchedPlan(asdBlasHandle handle, const int64_t n, const int64_t batchSize);

// get minimum workspace size
AspbStatus asdBlasGetWorkspaceSize(asdBlasHandle handle, size_t& workspaceSize);

// set workspace ptr
AspbStatus asdBlasSetWorkspace(asdBlasHandle handle, void* workSpace);

// Destroy a handle, free all spaces allocated for it.
AspbStatus asdBlasDestroy(asdBlasHandle handle);

// 同步
AspbStatus asdBlasSynchronize(asdBlasHandle handle);

// Executes a Blas op named ssyr2.
AspbStatus asdBlasSsyr2(asdBlasHandle handle, asdBlasFillMode_t uplo, const int64_t n, const float& alpha, aclTensor* x,
                        int64_t incx, aclTensor* y, int64_t incy, aclTensor* A, const int64_t lda);

AspbStatus asdBlasSsyr(asdBlasHandle handle, asdBlasFillMode_t uplo, const int64_t n, const float& alpha, aclTensor* x,
                       const int64_t incx, aclTensor* A, const int64_t lda);

AspbStatus asdBlasCgemm(asdBlasHandle handle, asdBlasOperation_t transa, asdBlasOperation_t transb, const int64_t m,
                        const int64_t n, const int64_t k, const std::complex<float>& alpha, aclTensor* A,
                        const int64_t lda, aclTensor* B, const int64_t ldb, const std::complex<float>& beta,
                        aclTensor* C, const int64_t ldc);

AspbStatus asdBlasStrmv(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag,
                        const int64_t n, aclTensor* A, const int64_t lda, aclTensor* x, const int64_t incx);

AspbStatus asdBlasCgemv(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m, const int64_t n,
                        const std::complex<float>& alpha, aclTensor* A, const int64_t lda, aclTensor* x,
                        const int64_t incx, const std::complex<float>& beta, aclTensor* y, const int64_t incy);

AspbStatus asdBlasCtrmv(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag,
                        const int64_t n, aclTensor* A, const int64_t lda, aclTensor* x, const int64_t incx);

AspbStatus asdBlasSasum(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result);

AspbStatus asdBlasScasum(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result);

AspbStatus asdBlasSnrm2(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result);

AspbStatus asdBlasScnrm2(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result);

AspbStatus asdBlasScopy(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy);

AspbStatus asdBlasCcopy(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy);

AspbStatus asdBlasSscal(asdBlasHandle handle, const int64_t n, const float& alpha, aclTensor* x, const int64_t incx);

AspbStatus asdBlasCsscal(asdBlasHandle handle, const int64_t n, const float& alpha, aclTensor* x, const int64_t incx);

AspbStatus asdBlasCscal(asdBlasHandle handle, const int64_t n, const std::complex<float>& alpha, aclTensor* x,
                        const int64_t incx);

AspbStatus asdBlasSdot(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                       const int64_t incy, aclTensor* result);

AspbStatus asdBlasCdotu(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy, aclTensor* result);

AspbStatus asdBlasCdotc(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy, aclTensor* result);

AspbStatus asdBlasColwiseMul(asdBlasHandle handle, const int64_t m, const int64_t n, aclTensor* mat, aclTensor* vec,
                             aclTensor* result);

AspbStatus asdBlasComplexMatDot(asdBlasHandle handle, const int64_t m, const int64_t n, aclTensor* matx,
                                aclTensor* maty, aclTensor* result);

AspbStatus asdBlasCgerc(asdBlasHandle handle, const int64_t m, const int64_t n, const std::complex<float>& alpha,
                        aclTensor* x, const int64_t incx, aclTensor* y, const int64_t incy, aclTensor* A,
                        const int64_t lda);

AspbStatus asdBlasIcamax(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result);

AspbStatus asdBlasIsamax(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result);

AspbStatus asdBlasCaxpy(asdBlasHandle handle, const int64_t n, const std::complex<float>& alpha, aclTensor* x,
                        int64_t incx, aclTensor* y, int64_t incy);

AspbStatus asdBlasSswap(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy);

AspbStatus asdBlasCswap(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy);

AspbStatus asdBlasStrmm(asdBlasHandle handle, asdBlasSideMode_t side, asdBlasFillMode_t uplo, asdBlasOperation_t trans,
                        asdBlasDiagType_t diag, const int64_t m, const int64_t n, const float& alpha, aclTensor* A,
                        const int64_t lda, aclTensor* B, const int64_t ldb, aclTensor* C, const int64_t ldc);

AspbStatus asdBlasCsrot(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy, const float& c, const float& s);

AspbStatus asdBlasHCgemvBatched(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m, const int64_t n,
                                const std::complex<op::fp16_t>& alpha, aclTensor* A, const int64_t lda, aclTensor* x,
                                const int64_t incx, const std::complex<op::fp16_t>& beta, aclTensor* y,
                                const int64_t incy, const int64_t batchCount);

AspbStatus asdBlasCgemvBatched(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m, const int64_t n,
                               const std::complex<float>& alpha, aclTensor* A, const int64_t lda, aclTensor* x,
                               const int64_t incx, const std::complex<float>& beta, aclTensor* y, const int64_t incy,
                               const int64_t batchCount);

AspbStatus asdBlasHCgemmBatched(asdBlasHandle handle, asdBlasOperation_t transa, asdBlasOperation_t transb,
                                const int64_t m, const int64_t n, const int64_t k,
                                const std::complex<op::fp16_t>& alpha, aclTensor* A, const int64_t lda, aclTensor* B,
                                const int64_t ldb, const std::complex<op::fp16_t>& beta, aclTensor* C,
                                const int64_t ldc, const int64_t batchCount);

AspbStatus asdBlasCgemmBatched(asdBlasHandle handle, asdBlasOperation_t transa, asdBlasOperation_t transb,
                               const int64_t m, const int64_t n, const int64_t k, const std::complex<float>& alpha,
                               aclTensor* A, const int64_t lda, aclTensor* B, const int64_t ldb,
                               const std::complex<float>& beta, aclTensor* C, const int64_t ldc,
                               const int64_t batchCount);

AspbStatus asdBlasHCmatinvBatched(asdBlasHandle handle, const int64_t n, aclTensor* A, const int64_t lda,
                                  aclTensor* Ainv, const int64_t lda_inv, aclTensor* info, int64_t batchSize);

AspbStatus asdBlasCmatinvBatched(asdBlasHandle handle, const int64_t n, aclTensor* A, const int64_t lda,
                                 aclTensor* Ainv, const int64_t lda_inv, aclTensor* info, int64_t batchSize);
// batched real-matrix inversion, float32, shape [batchSize, n, n]
AspbStatus asdBlasSmatinvBatched(asdBlasHandle handle, const int64_t n, aclTensor* A, const int64_t lda,
                                 aclTensor* Ainv, const int64_t lda_inv, aclTensor* info, int64_t batchSize);

// batched real-matrix inversion, float16, shape [batchSize, n, n]
AspbStatus asdBlasHSmatinvBatched(asdBlasHandle handle, const int64_t n, aclTensor* A, const int64_t lda,
                                  aclTensor* Ainv, const int64_t lda_inv, aclTensor* info, int64_t batchSize);

} // namespace AsdSip
#endif
