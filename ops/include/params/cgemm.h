/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_CGEMM_H
#define ASDSIP_PARAMS_CGEMM_H

#include <cstdint>
#include <string>
#include <sstream>
#include <complex>

namespace AsdSip {
namespace OpParam {
struct Cgemm {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t transA;
    int64_t transB;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldaPad;
    int64_t ldbPad;

    std::complex<float> alpha;
    std::complex<float> beta;

    bool operator==(const Cgemm& other) const
    {
        return this->m == other.m && this->n == other.n && this->k == other.k && this->transA == other.transA &&
               this->transB == other.transB && this->lda == other.lda && this->ldb == other.ldb &&
               this->ldc == other.ldc && this->ldaPad == other.ldaPad && this->ldbPad == other.ldbPad;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "alpha:" << alpha << ", beta:" << beta << ", m:" << m << ", n:" << n << ", k:" << k
           << ", transA:" << transA << ", transB:" << transB << ", lda:" << lda << ", ldb:" << ldb << ", ldc:" << ldc
           << ", ldaPad:" << ldaPad << ", ldbPad:" << ldbPad;
        return ss.str();
    }
};
} // namespace OpParam
} // namespace AsdSip

#endif // ASDSIP_PARAMS_CGEMM_H
