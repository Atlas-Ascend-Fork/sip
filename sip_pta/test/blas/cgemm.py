#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import logging
import random
import sys
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch_npu  # noqa: F401  # torch.npu 属性由 torch_npu 导入时注册，不可移除
import torch_sip

# 配置全局日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("torch_sip_cgemm")


@dataclass(frozen=True)
class GemmConfig:
    """参数对象：封装 GEMM 的所有维度和模式配置"""

    m: int
    n: int
    k: int
    trans_a: str = "N"
    trans_b: str = "N"
    alpha: Union[complex, float] = 1.0
    beta: Union[complex, float] = 0.0
    name: str = "Case"


class CgemmComprehensiveTester:
    def __init__(self, device: str = "npu:0"):
        self.device = device

    @staticmethod
    def get_random_scalar() -> Union[complex, float]:
        """生成随机标量逻辑"""
        real = random.uniform(-2, 2)
        if random.random() > 0.5:
            imag = random.uniform(-2, 2)
            return complex(real, imag)
        return real

    @staticmethod
    def to_column_major(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.t().contiguous().t()

    def get_complex_mat(self, shape: Tuple[int, ...]) -> torch.Tensor:
        real = torch.randn(shape, dtype=torch.float32)
        imag = torch.randn(shape, dtype=torch.float32)
        return torch.complex(real, imag).to(self.device)

    def run_case(self, cfg: GemmConfig) -> bool:
        # 1. 数据准备 (利用 cfg 属性)
        shape_a = (cfg.k, cfg.m) if cfg.trans_a.upper() != "N" else (cfg.m, cfg.k)
        shape_b = (cfg.n, cfg.k) if cfg.trans_b.upper() != "N" else (cfg.k, cfg.n)

        mat_a = self.get_complex_mat(shape_a)
        mat_b = self.get_complex_mat(shape_b)
        mat_c = self.get_complex_mat((cfg.m, cfg.n))
        mat_c_init = mat_c.clone()

        a_in = self.to_column_major(mat_a)
        b_in = self.to_column_major(mat_b)
        c_in = self.to_column_major(mat_c)

        # 2. 核心调用
        try:
            torch_sip.asd_blas_cgemm(
                a_in, b_in, c_in, cfg.alpha, cfg.beta, cfg.trans_a, cfg.trans_b
            )
        except Exception as exc:
            logger.error("[%s] Operator failed: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算
        def apply_op(m, op):
            op = op.upper()
            return m.t() if op == "T" else (m.conj().t() if op == "C" else m)

        a_ref = apply_op(mat_a.cpu(), cfg.trans_a)
        b_ref = apply_op(mat_b.cpu(), cfg.trans_b)
        ref = cfg.alpha * torch.matmul(a_ref, b_ref) + cfg.beta * mat_c_init.cpu()

        # 4. 比对
        is_close = torch.allclose(c_in.cpu(), ref, rtol=1e-3, atol=1e-3)
        if not is_close:
            logger.error("[FAIL] %s | Dim: %dx%dx%d", cfg.name, cfg.m, cfg.n, cfg.k)

        return is_close


def main():
    tester = CgemmComprehensiveTester()
    iteration = 0
    pass_count = 0

    while iteration < 100:
        iteration += 1

        # 使用辅助方法生成随机参数，保持 main 函数整洁
        m, n, k = (random.randint(1, 64) for _ in range(3))

        # 构造配置对象 (解决参数过多问题)
        config = GemmConfig(
            m=m,
            n=n,
            k=k,
            alpha=tester.get_random_scalar(),
            beta=tester.get_random_scalar(),
            trans_a=random.choice(["N", "T", "C"]),
            trans_b=random.choice(["N", "T", "C"]),
            name=f"Iter_{iteration}",
        )

        if tester.run_case(config):
            pass_count += 1

        if (iteration - 1) % 100 == 0:
            logger.info("Progress: %d/100 cases finished.", iteration)

    # 枚举域与形状一致性负向用例（issue #386）：非法 trans、非 2 维 A、B/C 与
    # m/n/k 不匹配均应在绑定层报参数错误，不再静默下传内核。
    # 包装层 trans 入参为字符串（"N"/"T"/"C"），非法枚举分两层验证：
    # 包装层非法字符串报 ValueError；绕过包装层直连 torch.ops 传越界 int
    # 触达 C++ 侧 TORCH_CHECK 报 RuntimeError（int 入参走包装层会先在
    # _get_trans_enum 处 AttributeError，测不到 C++ 校验）
    logger.info("-" * 60)
    logger.info("Enum/shape negative cases (issue #386):")
    mat_a = tester.get_complex_mat((4, 8))
    mat_b = tester.get_complex_mat((8, 6))
    mat_c = tester.get_complex_mat((4, 6))
    a_in = tester.to_column_major(mat_a)
    b_in = tester.to_column_major(mat_b)
    c_in = tester.to_column_major(mat_c)

    all_intercepted = True

    # 包装层枚举域：非法字符串在包装层报 ValueError
    try:
        torch_sip.asd_blas_cgemm(a_in, b_in, c_in, 1.0, 0.0, "X", "N")
    except ValueError as exc:
        logger.info(
            "  [PASS] invalid trans_a='X' (wrapper) 拦截成功: %s...", str(exc)[:60]
        )
    else:
        logger.error("  [FAIL] invalid trans_a='X' 未被拦截!")
        all_intercepted = False

    # C++ 侧枚举域：直连 torch.ops 传越界 int，触达 C++ TORCH_CHECK
    for msg, t_a, t_b in [
        ("invalid transA=42 (torch.ops)", 42, 0),
        ("invalid transB=42 (torch.ops)", 0, 42),
    ]:
        try:
            torch.ops.torch_sip.asd_blas_cgemm(a_in, b_in, c_in, 1.0, 0.0, t_a, t_b)
        except RuntimeError as exc:
            logger.info("  [PASS] %s 拦截成功: %s...", msg, str(exc)[:60])
        else:
            logger.error("  [FAIL] %s 未被拦截!", msg)
            all_intercepted = False

    # 形状不匹配：合法枚举经包装层下传，触达 C++ 形状校验
    negative_shape_suites = [
        ("3D A", tester.get_complex_mat((2, 4, 8)), mat_b, mat_c),
        ("B k-dim mismatch", mat_a, tester.get_complex_mat((16, 6)), mat_c),
        ("C shape mismatch", mat_a, mat_b, tester.get_complex_mat((4, 8))),
    ]
    for msg, a_neg, b_neg, c_neg in negative_shape_suites:
        try:
            torch_sip.asd_blas_cgemm(a_neg, b_neg, c_neg, 1.0, 0.0, "N", "N")
        except RuntimeError as exc:
            logger.info("  [PASS] %s 拦截成功: %s...", msg, str(exc)[:60])
        else:
            logger.error("  [FAIL] %s 未被拦截!", msg)
            all_intercepted = False

    return 0 if pass_count == iteration and all_intercepted else 1


if __name__ == "__main__":
    sys.exit(main())
