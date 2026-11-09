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
logger = logging.getLogger("torch_sip_cgerc_test")


@dataclass(frozen=True)
class GercConfig:
    """参数对象：封装 CGERC 的所有维度与标量配置"""

    m: int
    n: int
    alpha: Union[complex, float]
    name: str = "Case"


class CgercTester:
    """CGERC 算子功能测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device

    @staticmethod
    def to_column_major(tensor: torch.Tensor) -> torch.Tensor:
        """确保矩阵满足列优先布局需求 (静态方法)"""
        if tensor.dim() == 2:
            return tensor.t().contiguous().t()
        return tensor.contiguous()

    def get_complex_tensor(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """安全生成复数张量 (先 CPU 后搬移，避开 NPU 随机数限制)"""
        real = torch.randn(shape, dtype=torch.float32)
        imag = torch.randn(shape, dtype=torch.float32)
        return torch.complex(real, imag).to(self.device)

    def run_case(self, cfg: GercConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据与输入
        mat_a = self.get_complex_tensor((cfg.m, cfg.n))
        vec_x = self.get_complex_tensor((cfg.m,))
        vec_y = self.get_complex_tensor((cfg.n,))

        # 备份初始数据用于 CPU 参考计算
        mat_a_ref = mat_a.cpu()
        vec_x_cpu = vec_x.cpu()
        vec_y_cpu = vec_y.cpu()

        a_in = mat_a.contiguous()
        x_in = vec_x.contiguous()
        y_in = vec_y.contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹算子核心逻辑)
        try:
            torch_sip.asd_blas_cgerc(a_in, x_in, y_in, cfg.alpha)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算: A = alpha * outer(x, conj(y)) + A
        ref = (cfg.alpha * torch.outer(vec_x_cpu, vec_y_cpu.conj())) + mat_a_ref

        # 4. 比对验证
        is_close = torch.allclose(a_in.cpu(), ref, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        logger.info(
            "[%s] %-20s | M=%d, N=%d | Alpha=%s",
            status,
            cfg.name,
            cfg.m,
            cfg.n,
            cfg.alpha,
        )

        if not is_close:
            max_diff = torch.abs(a_in.cpu() - ref).max()
            logger.error("      Max Diff: %.6e", max_diff)

        return is_close


def main():
    """主测试套件入口"""
    tester = CgercTester()
    logger.info("开始 CGERC 专项测试 ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        GercConfig(8, 4, 1.0, "Real_Alpha"),
        GercConfig(16, 16, complex(0.5, 0.5), "Complex_Alpha"),
        GercConfig(128, 64, 1.0, "Large_Rectangle"),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    # 维度一致性负向用例（issue #372）：1D A 应在绑定层报参数错误而非底层 IndexError
    logger.info("-" * 60)
    logger.info("Dimension negative cases (issue #372):")
    a_1d = tester.get_complex_tensor((8,))
    x = tester.get_complex_tensor((8,))
    y = tester.get_complex_tensor((4,))
    try:
        torch_sip.asd_blas_cgerc(a_1d, x, y, 1.0)
        logger.error("  - 1D A: FAILED | wrong-dim input was accepted")
        all_passed = False
    except RuntimeError as exc:
        if "cgerc" not in str(exc) and "2D" not in str(exc):
            logger.error("  - 1D A: FAILED | unexpected error: %s", exc)
            all_passed = False
        else:
            logger.info("  - 1D A: PASSED | %s", exc)

    logger.info("-" * 60)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    # 整个脚本唯一的 sys.exit 出口
    sys.exit(main())
