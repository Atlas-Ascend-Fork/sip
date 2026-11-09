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
import sys
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_cgemv_test")


@dataclass(frozen=True)
class GemvConfig:
    """参数对象：封装 CGEMV 的所有维度、标量及模式配置"""

    m: int
    n: int
    alpha: Union[complex, float]
    beta: Union[complex, float]
    trans: str = "N"
    name: str = "Case"


class CgemvTester:
    """CGEMV 算子功能测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device

    @staticmethod
    def to_column_major(tensor: torch.Tensor) -> torch.Tensor:
        """确保矩阵满足列优先布局需求"""
        if tensor.dim() == 2:
            return tensor.t().contiguous().t()
        return tensor.contiguous()

    def get_complex_tensor(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """安全生成复数张量 (先 CPU 后搬移)"""
        real = torch.randn(shape, dtype=torch.float32)
        imag = torch.randn(shape, dtype=torch.float32)
        return torch.complex(real, imag).to(self.device)

    def run_case(self, cfg: GemvConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块，仅包裹算子核心逻辑。
        """
        # 1. 确定逻辑维度并构造数据
        is_trans = cfg.trans.upper() != "N"
        x_len, y_len = (cfg.m, cfg.n) if is_trans else (cfg.n, cfg.m)

        mat_a = self.get_complex_tensor((cfg.m, cfg.n))
        vec_x = self.get_complex_tensor((x_len,))
        vec_y = self.get_complex_tensor((y_len,))
        vec_y_init = vec_y.clone()

        # 转换为算子库要求的布局
        a_in = self.to_column_major(mat_a)
        x_in = vec_x.contiguous()
        y_in = vec_y.contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹 NPU 算子执行)
        try:
            torch_sip.asd_blas_cgemv(a_in, x_in, y_in, cfg.alpha, cfg.beta, cfg.trans)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块以明确异常源)
        def apply_op(mat, op):
            op = op.upper()
            if op == "T":
                return mat.t()
            if op == "C":
                return mat.conj().t()
            return mat

        a_ref = apply_op(mat_a.cpu(), cfg.trans)
        ref = torch.addmv(
            vec_y_init.cpu() * cfg.beta, a_ref, vec_x.cpu(), alpha=cfg.alpha, beta=1.0
        )

        # 4. 比对验证
        is_close = torch.allclose(y_in.cpu(), ref, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        logger.info(
            "[%s] %-25s | M=%d, N=%d | Trans=%s",
            status,
            cfg.name,
            cfg.m,
            cfg.n,
            cfg.trans,
        )

        if not is_close:
            logger.error("      Max Diff: %.6e", (y_in.cpu() - ref).abs().max())

        return is_close


def main():
    """主测试套件入口"""
    tester = CgemvTester()
    logger.info("开始 CGEMV 专项测试 (参数对象化重构版本) ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        GemvConfig(8, 4, 1.0, 0.0, "N", "Normal"),
        GemvConfig(8, 8, complex(1.0, 0.5), 0.0, "N", "Complex_Alpha"),
        GemvConfig(8, 8, 1.0, 1.0, "T", "Transpose"),
        GemvConfig(320, 320, 1.0, 0.0, "N", "Large_Size"),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    # 维度一致性负向用例（issue #372）：3D A 应在绑定层报参数错误而非错误维度语义下传
    logger.info("-" * 60)
    logger.info("Dimension negative cases (issue #372):")
    a_3d = tester.get_complex_tensor((2, 8, 8))
    x = tester.get_complex_tensor((8,))
    y = tester.get_complex_tensor((8,))
    try:
        torch_sip.asd_blas_cgemv(a_3d, x, y, 1.0, 0.0, "N")
        logger.error("  - 3D A: FAILED | wrong-dim input was accepted")
        all_passed = False
    except RuntimeError as exc:
        if "cgemv" not in str(exc) and "2D" not in str(exc):
            logger.error("  - 3D A: FAILED | unexpected error: %s", exc)
            all_passed = False
        else:
            logger.info("  - 3D A: PASSED | %s", exc)

    logger.info("-" * 60)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
