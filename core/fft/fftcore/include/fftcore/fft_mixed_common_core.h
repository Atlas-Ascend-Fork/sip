/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "fftoperation/fft_operation.h"
#include "params/fft_mixed_common.h"
class FftMixedCommonCore : public FftOperation {
public:
    explicit FftMixedCommonCore(AsdSip::OpParam::FftMixedCommon param);
    ~FftMixedCommonCore() override;
    bool init() override;
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor&, Tensor&, void*, workspace::Workspace&) override;
    void Run(void*, void*, void*, workspace::Workspace&) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
