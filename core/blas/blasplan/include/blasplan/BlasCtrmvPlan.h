/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "BlasPlan.h"

namespace AsdSip {
class BlasCtrmvPlan : public BlasPlan {
public:
    explicit BlasCtrmvPlan(asdBlasFillMode_t uplo, int64_t n);
    ~BlasCtrmvPlan() override;
    AsdSip::AspbStatus CreateTensor() override;
    AsdSip::AspbStatus SetUploTensor();
    AsdSip::AspbStatus FreeTensor() override;
    Mki::Tensor GetUploTensor() { return this->uploTensor; }
    aclTensor* GetUploAclTensor() { return this->uploAclTensor; }
    asdBlasFillMode_t GetUplo() const { return uplo; }
    int64_t GetN() const { return n; }
    int64_t GetWorkspaceSize() override;

private:
    Mki::Tensor uploTensor;
    aclTensor* uploAclTensor = nullptr;
    asdBlasFillMode_t uplo;
    int64_t n;
};
} // namespace AsdSip
