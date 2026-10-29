/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "pytorch_npu_helper.hpp"

#include "base_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace sip_pta {

/**
 * @brief 批量线性方程组求解 A·X = B (fp32, n∈[2,256])
 *
 * @param A 系数矩阵 [batch, n, n]
 * @param B 右端项 [batch, n, nrhs]
 * @return X 解 [batch, n, nrhs]
 */
at::Tensor solve(const at::Tensor& A, const at::Tensor& B)
{
    TORCH_CHECK(A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "solve: only float32 supported in current version.");

    TORCH_CHECK(A.dim() == 3 && A.size(1) == A.size(2), "solve: A must be [batch, n, n], got dim ", A.dim());

    TORCH_CHECK(B.dim() == 3 && B.size(0) == A.size(0) && B.size(1) == A.size(1),
                "solve: B must be [batch, n, nrhs] with same batch/n as A.");

    c10_npu::NPUGuard guard(A.device());

    at::Tensor X = at::empty_like(B);
    auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);

    size_t workspace_size = 0;
    AsdSip::solveGetWorkspaceSize(workspace_size);
    at::Tensor workspace_tensor;
    void* workspace_addr = nullptr;
    if (workspace_size != 0) {
        workspace_tensor = at_npu::native::allocate_workspace(static_cast<long>(workspace_size), sip_stream);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    aclTensor* acl_a = CreateAclTensorFromAtTensor(A);
    aclTensor* acl_b = CreateAclTensorFromAtTensor(B);
    aclTensor* acl_x = CreateAclTensorFromAtTensor(X);
    EXEC_FUNC(AsdSip::solve, acl_a, acl_b, acl_x, sip_stream, workspace_addr);
    return X;
}

// 算子注册 (前向: solve_forward, 与 solve_backward 对齐命名)
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("solve_forward(Tensor A, Tensor B) -> Tensor"); }

// 硬件后端实现绑定 (NPU 为 PrivateUse1)
TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("solve_forward", &solve); }

/**
 * @brief solve 反向: gB = solve(Aᵀ, gX), gA = -(gB @ Xᵀ) (fp32)
 *
 * 组合方案 (参照 smatinv_backward): 融合 solve 解 Aᵀ, gA 按矩阵阶数分派
 *   - n≤6 且中间张量 ≤128M 元素: 广播乘 + 末维归约(向量路径), 避免细长 GEMM 的
 *     Cube 分块低效(P1 广播 [196,3,16384] gA 步 5016µs → 272µs); 超限回落 bmm
 *   - 其余: ATen bmm (XT 拷贝 + bmm + negate)
 *
 * @param A  系数矩阵 [batch, n, n] (前向输入)
 * @param X  前向解 [batch, n, nrhs]
 * @param gX 上游梯度 [batch, n, nrhs]
 * @return (gB [batch, n, nrhs], gA [batch, n, n])
 */
std::tuple<at::Tensor, at::Tensor> solve_backward(const at::Tensor& A, const at::Tensor& X, const at::Tensor& gX)
{
    TORCH_CHECK(A.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat && gX.scalar_type() == at::kFloat,
                "solve_backward: only float32 supported in current version.");

    c10_npu::NPUGuard guard(A.device());

    // gB = solve(Aᵀ, gX)
    at::Tensor AT = A.transpose(-2, -1).contiguous();
    at::Tensor gB = sip_pta::solve(AT, gX);

    // gA = -(gB @ Xᵀ) = -Σ_r gB[b,i,r]·X[b,j,r]; 分派阈值: n≤6 (bmm 于大 n 更优),
    // nrhs≥2 (K=1 时 bmm 占优), 元素量 128M (中间张量物化上限)
    constexpr int64_t MULSUM_MAX_N = 6;
    constexpr int64_t MULSUM_MAX_ELEMS = 128LL * 1024 * 1024;
    const int64_t batch = A.size(0);
    const int64_t n = A.size(1);
    const int64_t nrhs = X.size(2);
    at::Tensor gA;
    if (n <= MULSUM_MAX_N && nrhs >= 2 && batch * n * n * nrhs <= MULSUM_MAX_ELEMS) {
        // 向量路径: [b,n,1,nrhs] × [b,1,n,nrhs] 广播乘 → 末维 Sum 归约
        at::Tensor prod = at::mul(gB.unsqueeze(2), X.unsqueeze(1));
        gA = at::sum(prod, std::vector<int64_t>{-1});
        gA = -gA;
    } else {
        // Cube 路径: bmm + negate
        // 注: baddbmm 的 self 需与输出同形, 而 gB/X 的 nrhs 维 ≠ n 不可广播, 无法以 alpha 折叠负号
        at::Tensor XT = X.transpose(-2, -1).contiguous();
        gA = at::bmm(gB, XT);
        gA = -gA;
    }

    return std::make_tuple(gB, gA);
}

TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("solve_backward(Tensor A, Tensor X, Tensor gX) -> (Tensor gB, Tensor gA)");
}
TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("solve_backward", &solve_backward); }
} // namespace sip_pta
