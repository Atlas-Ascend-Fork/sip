/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTOPERATION_TRANSPOSE__
#define __FFTOPERATION_TRANSPOSE__

#include "fftoperation/fft_operation.h"

class Transpose : public FftOperation {
public:
    Transpose(int axis0, int axis1, Mki::SVector<int64_t> dims);
    ~Transpose() override;
    bool init() override;
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override;
    void Run(void* input, void* output, void* stream, workspace::Workspace& workspace) override;
    size_t EstimateWorkspaceSize() override;

protected:
    bool TryInitTiled();
    bool RunTiled(void* input, void* output, void* stream);
    size_t tiledBytes_ = 0;
    int axis0_;
    int axis1_;
    Mki::SVector<int64_t> dims_;
};

#endif
