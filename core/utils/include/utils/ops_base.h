/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_OPS_BASE_H
#define ASDSIP_OPS_BASE_H

#include <mki/utils/rt/rt.h>
#include "op_desc.h"
#include "mki/kernel.h"
#include "mki/launch_param.h"
#include "mki/tensor.h"
#include "mki/utils/status/status.h"
#include "mki/utils/SVector/SVector.h"
#include "mem_base_inner.h"

constexpr size_t ASYNC_WORKSPACE_SIZE = 2048;

Mki::Status RunAsdOps(MkiRtStream stream, const AsdSip::OpDesc& opDesc, const Mki::SVector<Mki::Tensor>& inTensorList,
                      Mki::SVector<Mki::Tensor>& outTensorList, uint8_t* workspace = nullptr);

Mki::Status RunAsdOpsV2(MkiRtStream stream, const AsdSip::OpDesc& opDesc, const Mki::SVector<aclTensor*>& inTensorList,
                        Mki::SVector<aclTensor*>& outTensorList, uint8_t* workspace = nullptr);

// Reuse a caller-owned initialized zero-workspace kernel. The caller must
// ensure that its metadata and device context match every invocation.
Mki::Status RunAsdOpsV2(MkiRtStream stream, const Mki::LaunchParam& launch, uint8_t* workspace,
                        Mki::Kernel& preparedKernel);
#endif
