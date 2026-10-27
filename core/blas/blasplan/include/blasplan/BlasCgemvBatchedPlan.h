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
using namespace Mki;

enum class asdDataType_t { ASD_C_32F = 0, ASD_C_64F };

class BlasCgemvBatchedPlan : public BlasPlan {
public:
    BlasCgemvBatchedPlan(asdBlasOperation_t trans, asdDataType_t dtype, const int64_t m);
    ~BlasCgemvBatchedPlan() override;
    AsdSip::AspbStatus CreateTensor() override;
    void SetMaskTensor();
    AsdSip::AspbStatus FreeTensor() override;
    uint32_t GetMaxMatNum() const { return this->maxMatNum; }
    // plan 绑定 m 访问器（issue #284）：mask 张量按 MakePlan 时的 m 推导尺寸，
    // 执行入口据此校验执行 m 与 plan m 一致，防止以更大执行 m 复用旧 plan 时
    // 内核 UB 按执行 m 展开越过按 plan m 计算的掩码/UB 边界
    int64_t GetPlanM() const { return this->m; }
    asdBlasOperation_t GetPlanTrans() const { return this->trans; }
    asdDataType_t GetPlanDtype() const { return this->dtype; }
    aclTensor* GetAclMaskTensor() { return this->maskAclTensor; }

private:
    Tensor maskTensor;
    aclTensor* maskAclTensor = nullptr;
    asdBlasOperation_t trans;
    asdDataType_t dtype;
    int64_t m;
    uint32_t maxMatNum;
};
} // namespace AsdSip
