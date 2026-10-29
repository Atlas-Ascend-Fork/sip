/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_BASE_INNER_API_H
#define ASDSIP_BASE_INNER_API_H

#include "mki/tensor.h"
#include "mki/types.h"
#include "utils/aspb_status.h"

namespace AsdSip {
AsdSip::AspbStatus Conj(const Mki::Tensor& inTensor, Mki::Tensor& outTensor, void* stream,
                        uint8_t* workspace = nullptr);

AsdSip::AspbStatus MatMul(const Mki::Tensor& inATensor, const Mki::Tensor& inBTensor, Mki::Tensor& outTensor, int m,
                          int k, int n, void* stream, uint8_t* deviceBuffer = nullptr);

AsdSip::AspbStatus AsStrided(const Mki::Tensor& inTensor, Mki::Tensor& outTensor, Mki::SVector<int64_t> size,
                             Mki::SVector<int64_t> stride, void* stream, int64_t offset = 0,
                             uint8_t* deviceBuffer = nullptr);

AsdSip::AspbStatus Slice(const Mki::Tensor& inTensor, Mki::Tensor& outTensor, int64_t dim, int64_t start, int64_t end,
                         int64_t step, void* stream, uint8_t* deviceBuffer = nullptr);

AsdSip::AspbStatus Reverse(const Mki::Tensor& inTensor, Mki::Tensor& outTensor, Mki::SVector<int64_t> axis,
                           void* stream, uint8_t* deviceBuffer = nullptr);

AsdSip::AspbStatus Concat(const Mki::Tensor& inTensorA, const Mki::Tensor& inTensorB, Mki::Tensor& outTensor,
                          int concatDim, void* stream, uint8_t* deviceBuffer = nullptr);

AsdSip::AspbStatus Mul(const Mki::Tensor& inputA, const Mki::Tensor& inTensorB, Mki::Tensor& outTensor, void* stream,
                       uint8_t* deviceBuffer);

AsdSip::AspbStatus Div(const Mki::Tensor& inTensorA, const Mki::Tensor& inTensorB, Mki::Tensor& outTensor, void* stream,
                       uint8_t* deviceBuffer = nullptr);

AsdSip::AspbStatus UnfoldGrad(const Mki::Tensor& inTensor, Mki::Tensor& outTensor, Mki::SVector<int64_t> sizes,
                              int64_t dim, int64_t size, int64_t step, void* stream, uint8_t* deviceBuffer = nullptr);

AsdSip::AspbStatus MakeComplex(const Mki::Tensor& realTensor, const Mki::Tensor& imagTensor, Mki::Tensor& outTensor,
                               int64_t dim, void* stream, uint8_t* deviceBuffer = nullptr);

AsdSip::AspbStatus Muls(const Mki::Tensor& inTensorA, float value, Mki::Tensor& outTensor, void* stream,
                        uint8_t* deviceBuffer = nullptr);

} // namespace AsdSip

#endif
