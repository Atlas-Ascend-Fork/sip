/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * torch_sip.npu_inv 最终 PTA 实现 (自动反向 + 手动反向两路径):
 *
 *   smatinv_inv_forward  : 纯前向, EXEC_BLAS_FUNC 直调 SiP asdBlasSmatinvBatched/HSmatinvBatched
 *                          (Gauss-Jordan 单 kernel, fp32 30us / fp16, [B,N,N] N<32, B<=3000)
 *   smatinv_inv_autograd : 自动反向入口 (npu_inv 内部调用, requires_grad 时)。它在 C++ 层通过
 *                          torch::autograd::Function 向引擎注册反向节点 (等价 torch 原生算子的
 *                          "前向 + 注册 backward 公式"); 用户侧照常 y.backward(dy), autograd
 *                          引擎回调 C++ backward, 全程无 python 参与。
 *   smatinv_backward     : 纯反向 dA = -(Y^T dY) Y^T (手动反向路径, python npu_inv_backward 入口)
 *
 * 反向公式: dA = -(Y^T dY) Y^T = Y^T (-dY) Y^T (负号折叠, 2 kernel: bmm x2)
 */
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "pytorch_npu_helper.hpp"

#include "blas_api.h"

#include <acl/acl.h>
#include <tuple>

namespace sip_pta {

// ---------------- 前向: SiP kernel 直通 (fp32/fp16, no-grad) ----------------
// n < 32: Smatinv Gauss-Jordan 消元 kernel (fp32/fp16 原生)
// 32 <= n <= 256: SgetriBatched 真实数 LU 求逆 kernel (fp32; fp16 升 fp32 计算后回转)
at::Tensor smatinv_inv_forward(const at::Tensor& A)
{
    TORCH_CHECK((A.scalar_type() == at::kFloat || A.scalar_type() == at::kHalf),
                "smatinv_inv_forward: Tensor must be Float32 or Float16.");
    TORCH_CHECK(A.dim() == 3 && A.size(1) == A.size(2),
                "smatinv_inv_forward: Tensor must be a 3D square matrix [Batch, N, N].");

    c10_npu::NPUGuard guard(A.device());
    int64_t batchSize = A.size(0);
    int64_t n = A.size(1);
    int64_t lda = n;
    int64_t lda_inv = n;

    // fp16 大矩阵: Sgetri 为 fp32 kernel, 升精度计算后回转
    at::Tensor work = (A.scalar_type() == at::kHalf && n >= 32) ? A.to(at::kFloat) : A;
    at::Tensor Ainv = at::empty_like(work);
    at::Tensor info; // 算子定义占位, 不生效

    std::vector<int64_t> params = {n, batchSize};
    aclTensor* acl_a = CreateAclTensorFromAtTensor(work);
    aclTensor* acl_a_inv = CreateAclTensorFromAtTensor(Ainv);

    auto makePlan = [n, batchSize](AsdSip::asdBlasHandle handle) {
        return AsdSip::asdBlasMakeSmatinvBatchedPlan(handle, n, batchSize);
    };

    if (work.scalar_type() == at::kFloat) {
        EXEC_BLAS_FUNC(AsdSip::asdBlasSmatinvBatched, makePlan, params, n, acl_a, lda, acl_a_inv, lda_inv, info,
                       batchSize);
    } else {
        EXEC_BLAS_FUNC(AsdSip::asdBlasHSmatinvBatched, makePlan, params, n, acl_a, lda, acl_a_inv, lda_inv, info,
                       batchSize);
    }

    if (A.scalar_type() == at::kHalf && n >= 32) {
        Ainv = Ainv.to(at::kHalf);
    }
    return Ainv;
}

// ---------------- 反向: dA = -(Y^T dY) Y^T (负号折叠进 GEMM epilogue, fp32 域) ----------------
at::Tensor smatinv_backward(const at::Tensor& Y, const at::Tensor& gradY)
{
    const at::Tensor& y = Y;
    TORCH_CHECK(y.scalar_type() == at::kFloat && gradY.scalar_type() == at::kFloat,
                "smatinv_backward: only Float32 supported (fp16 入口由上层转 fp32).");
    TORCH_CHECK(y.dim() == 3 && y.size(1) == y.size(2) && gradY.sizes() == y.sizes(),
                "smatinv_backward: expect [B, N, N] for both inputs.");

    at::Tensor yT = y.transpose(-1, -2); // view, 零拷贝
    at::Tensor tmp = at::empty_like(y);
    at::Tensor dA = at::empty_like(y);
    // tmp = beta*gradY + alpha*(Y^T @ dY) = -(Y^T dY)
    // beta=0 时 aten 语义忽略 input; 即使实现按字面 0*gradY 计算, 有限值也得到 0,
    // 两种语义结果一致 —— 用 gradY 当 dummy input 规避未初始化内存 NaN*0 的坑,
    // 同时省掉独立 neg kernel 与 dyNeg 临时内存 (3 kernel -> 2 kernel)。
    at::baddbmm_out(tmp, gradY, yT, gradY, 0.0, -1.0);
    at::bmm_out(dA, tmp, yT); // dA = -(Y^T dY) @ Y^T
    return dA;
}

// fp32/fp16 统一反向入口 (fp16 在 fp32 域计算后回转)
static at::Tensor smatinv_backward_any(const at::Tensor& y, const at::Tensor& gradY, at::ScalarType outType)
{
    if (y.scalar_type() == at::kFloat) {
        return smatinv_backward(y, gradY);
    }
    return smatinv_backward(y.to(at::kFloat), gradY.to(at::kFloat)).to(outType);
}

// ---------------- 自动反向: C++ autograd 节点 (AutogradPrivateUse1) ----------------
class SmatinvInvFunction : public torch::autograd::Function<SmatinvInvFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& A)
    {
        at::AutoDispatchBelowADInplaceOrView guard; // 防重入
        at::Tensor y = smatinv_inv_forward(A);
        ctx->save_for_backward({y});
        return y;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::tensor_list grad_output)
    {
        auto saved = ctx->get_saved_variables();
        return {smatinv_backward_any(saved[0], grad_output[0], saved[0].scalar_type())};
    }
};

at::Tensor smatinv_inv_autograd(const at::Tensor& A) { return SmatinvInvFunction::apply(A); }

// ---------------- 注册 (最终接口集) ----------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    // 纯前向 (no-grad 直通)
    m.def("smatinv_inv_forward(Tensor A) -> Tensor");
    // 纯反向: dA = -(Y^T dY) Y^T, Y 为前向输出
    m.def("smatinv_backward(Tensor Y, Tensor grad_y) -> Tensor");
    // 自动反向 (autograd 建图入口, npu_inv 内部调用)
    m.def("smatinv_inv_autograd(Tensor A) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
{
    m.impl("smatinv_inv_forward", &smatinv_inv_forward);
    m.impl("smatinv_backward", &smatinv_backward);
}

// 自动反向: autograd 键优先建图; no-grad 时走 PrivateUse1 直通
TORCH_LIBRARY_IMPL(torch_sip, AutogradPrivateUse1, m) { m.impl("smatinv_inv_autograd", &smatinv_inv_autograd); }
TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("smatinv_inv_autograd", &smatinv_inv_autograd); }
} // namespace sip_pta
