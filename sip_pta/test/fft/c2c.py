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
import time
import traceback
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
logger = logging.getLogger("torch_sip_fft_test")


@dataclass(frozen=True)
class FftConfig:
    """参数对象：封装 FFT 测试的形状、方向及变换维度"""

    shape: Tuple[int, ...]
    is_forward: bool = True
    transform_dims: int = 1
    name: str = "Case"


class FftTester:
    """快速傅里叶变换 (FFT) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        self.total_tests = 0
        self.passed_tests = 0

    @staticmethod
    def get_complex_data(shape: Tuple[int, ...], device: str) -> torch.Tensor:
        """生成 NPU 数据的复数张量"""
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)

    def run_dtype_negative_case(self) -> bool:
        """错误 dtype 负向用例（issue #371）：C2C 契约 complex64，
        float32/int 等错误 dtype 应在绑定层立即报错而非下传"""
        self.total_tests += 1
        for name, x in (
            ("float32", torch.randn(4, 8, dtype=torch.float32, device=self.device)),
            ("int32", torch.zeros(4, 8, dtype=torch.int32, device=self.device)),
        ):
            try:
                torch_sip.asd_fft_c2c(x, True)
                logger.error("  - %s: FAILED | wrong-dtype input was accepted", name)
                return False
            except RuntimeError as exc:
                if "complex64" not in str(exc):
                    logger.error("  - %s: FAILED | unexpected error: %s", name, exc)
                    return False
            logger.info("  - dtype negative (%s): PASSED", name)
        self.passed_tests += 1
        return True

    def validate(
        self, cfg: FftConfig, npu_out: torch.Tensor, ref_out: torch.Tensor
    ) -> bool:
        """
        统一校验逻辑 (遵循 G.FNM.07: 处理同步异常与返回值)
        """
        self.total_tests += 1

        # 1. 检查算子返回值合法性
        if npu_out is None:
            logger.error("  ✗ %s: FAILED | Operator returned None", cfg.name)
            return False

        try:
            # 2. 搬回 CPU (处理可能的显存访问异常或同步失败)
            npu_cpu = npu_out.cpu()

            # 3. 精度校验
            is_close = torch.allclose(npu_cpu, ref_out, atol=1e-4, rtol=1e-4)

            direction_str = "Forward" if cfg.is_forward else "Inverse"
            label = f"{cfg.transform_dims}D {direction_str} FFT {cfg.shape}"

            if is_close:
                logger.info("  ✓ %s: PASSED", label)
                self.passed_tests += 1
                return True

            # 4. 失败明细记录
            max_diff = (npu_cpu - ref_out).abs().max()
            logger.error("  ✗ %s: FAILED | Max Diff: %.6e", label, max_diff.item())
            return False

        except RuntimeError as e:
            # 捕获 NPU 异步执行抛出的错误
            logger.error(
                "  ⚠ %s: EXCEPTION | Validation failed during sync: %s", cfg.name, e
            )
            return False
        except Exception as e:
            logger.error("  ⚠ %s: UNKNOWN ERROR | %s", cfg.name, e)
            logger.debug(traceback.format_exc())
            return False

    def run_test_c2c(self, cfg: FftConfig):
        """通用 C2C FFT 测试接口"""
        # 1. 准备数据
        try:
            x_npu = self.get_complex_data(cfg.shape, self.device)
            x_cpu = x_npu.cpu()
        except Exception as e:
            logger.error("[%s] Data initialization failed: %s", cfg.name, e)
            self.total_tests += 1  # 计入总数但标记为未通过
            return

        # 2. 计算参考值
        dims = tuple(range(-cfg.transform_dims, 0))
        if cfg.is_forward:
            ref = torch.fft.fftn(x_cpu, dim=dims)
        else:
            # 逆向变换缩放处理
            scale = 1.0
            for d in dims:
                scale *= x_cpu.size(d)
            ref = torch.fft.ifftn(x_cpu, dim=dims) * scale

        # 3. 核心执行 (G.ERR.01: 最小化 try 块)
        out_npu = None
        duration = 0
        try:
            start_ts = time.time()
            out_npu = torch_sip.asd_fft_c2c(x_npu, cfg.is_forward)
            duration = (time.time() - start_ts) * 1000
        except Exception as exc:
            logger.error("[%s] Operator execution crashed: %s", cfg.name, exc)
            self.total_tests += 1
            return

        # 4. 结果校验并处理返回值 (G.FNM.07)
        success = self.validate(cfg, out_npu, ref)
        if success:
            logger.info("    - Execution Time: %.2f ms", duration)


def main():
    """测试套件入口"""
    if not torch.npu.is_available():
        logger.critical("NPU environment not detected.")
        return 1

    tester = FftTester()

    # 执行各维度测试
    test_cases = [
        FftConfig((16, 1024), is_forward=True, name="1D_Fwd"),
        FftConfig((16, 1024), is_forward=False, name="1D_Inv"),
        FftConfig((8, 4, 1024), is_forward=True, transform_dims=2, name="2D_Fwd"),
        FftConfig((8, 4, 1024), is_forward=False, transform_dims=2, name="2D_Inv"),
        FftConfig((2, 16, 32, 64), is_forward=True, transform_dims=3, name="3D_Fwd"),
        FftConfig((2, 16, 32, 64), is_forward=False, transform_dims=3, name="3D_Inv"),
    ]

    for case in test_cases:
        tester.run_test_c2c(case)

    # 错误 dtype 负向用例（issue #371）
    if not tester.run_dtype_negative_case():
        logger.error("dtype negative case failed")

    # 总结报告
    logger.info("\n" + "=" * 50)
    logger.info(
        "Final Report: %d/%d Tests Passed", tester.passed_tests, tester.total_tests
    )
    logger.info("=" * 50)

    # 遵循 G.FNM.07，确保脚本返回值反映测试结果
    return (
        0
        if (tester.passed_tests == tester.total_tests and tester.total_tests > 0)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
