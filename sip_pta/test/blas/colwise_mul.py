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
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_colwise_mul")


@dataclass(frozen=True)
class ColwiseMulConfig:
    """参数对象：封装矩阵按列乘法的维度配置"""

    m: int
    n: int
    name: str = "Case"


class ColwiseMulTester:
    """矩阵按列与向量逐元素相乘测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info(
            "当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env
        )

    @staticmethod
    def get_complex_tensor(shape: Tuple[int, ...], device: str) -> torch.Tensor:
        """
        静态方法：安全生成复数 Tensor。
        """
        real = torch.randn(shape, dtype=torch.float32)
        imag = torch.randn(shape, dtype=torch.float32)
        return torch.complex(real, imag).to(device)

    def run_case(self, cfg: ColwiseMulConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块，仅包裹算子核心逻辑。
        """
        # 1. 准备数据
        mat = self.get_complex_tensor((cfg.m, cfg.n), self.device)
        vec = self.get_complex_tensor((cfg.m,), self.device)

        # 备份用于 CPU 参考计算
        mat_cpu = mat.cpu()
        vec_cpu = vec.cpu()

        # 确保内存连续
        mat_in = mat.contiguous()
        vec_in = vec.contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹 NPU 算子执行)
        try:
            # 根据注册方式调用算子
            out_npu = torch_sip.asd_blas_colwise_mul(mat_in, vec_in)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块以明确异常源)
        # mat (m, n) * vec (m, 1) 利用广播机制实现按列相乘
        ref = mat_cpu * vec_cpu.unsqueeze(1)

        # 4. 比对验证
        out_cpu = out_npu.cpu()
        is_close = torch.allclose(out_cpu, ref, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-20s | M=%d, N=%d", status, cfg.name, cfg.m, cfg.n)

        if not is_close:
            max_diff = torch.abs(out_cpu - ref).max()
            logger.error("      Max Diff: %.6e", max_diff)

        return is_close


def main():
    """主测试套件入口"""
    tester = ColwiseMulTester()
    logger.info("开始 ColwiseMul 专项测试 ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        ColwiseMulConfig(8, 4, "Small_Matrix"),
        ColwiseMulConfig(16, 16, "Square_Matrix"),
        ColwiseMulConfig(128, 64, "Large_Rectangle"),
        ColwiseMulConfig(256, 1, "Single_Column"),
        ColwiseMulConfig(1, 256, "Single_Row"),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    # 维度一致性负向用例（issue #372）：1D mat / 2D vec 应在绑定层报参数错误
    logger.info("-" * 60)
    logger.info("Dimension negative cases (issue #372):")
    mat_1d = tester.get_complex_tensor((8,), tester.device)
    vec_2d = tester.get_complex_tensor((8, 4), tester.device)
    vec_ok = tester.get_complex_tensor((8,), tester.device)
    for name, (m_t, v_t) in (
        ("1D mat", (mat_1d, vec_ok)),
        ("2D vec", (vec_2d, vec_2d)),
    ):
        try:
            torch_sip.asd_blas_colwise_mul(m_t, v_t)
            logger.error("  - %s: FAILED | wrong-dim input was accepted", name)
            all_passed = False
        except RuntimeError as exc:
            if "colwise_mul" not in str(exc):
                logger.error("  - %s: FAILED | unexpected error: %s", name, exc)
                all_passed = False
            else:
                logger.info("  - %s: PASSED | %s", name, exc)

    logger.info("-" * 60)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    # 整个脚本唯一的退出出口
    sys.exit(main())
