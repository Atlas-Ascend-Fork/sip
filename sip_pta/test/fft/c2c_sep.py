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
logger = logging.getLogger("torch_sip_fft_sep")


@dataclass(frozen=True)
class FftSepConfig:
    """参数对象：封装分离输入模式 FFT 的维度、方向及预期状态"""

    shape: Tuple[int, ...]
    is_forward: bool = True
    transform_dims: int = 1
    expect_fail: bool = False
    name: str = "Case"


class FFTSesterSep:
    """分离输入模式 FFT 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        self.total_tests = 0
        self.passed_tests = 0

    @staticmethod
    def get_sep_data(
        shape: Tuple[int, ...], device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成分离实虚部（强约束内存连续）"""
        real = torch.randn(shape, dtype=torch.float32, device=device).contiguous()
        imag = torch.randn(shape, dtype=torch.float32, device=device).contiguous()
        return real, imag

    def run_dtype_negative_case(self) -> bool:
        """错误 dtype 负向用例（issue #371）：分离实虚部契约 float32，
        complex64/int 等错误 dtype 应在绑定层立即报错而非下传"""
        self.total_tests += 1
        real_c64 = torch.randn(4, 8, dtype=torch.float32, device=self.device).to(
            torch.complex64
        )
        imag_c64 = torch.randn(4, 8, dtype=torch.float32, device=self.device).to(
            torch.complex64
        )
        real_i32 = torch.zeros(4, 8, dtype=torch.int32, device=self.device)
        for name, (re_t, im_t) in (
            ("complex64", (real_c64, imag_c64)),
            ("int32", (real_i32, real_i32)),
        ):
            try:
                torch_sip.asd_fft_c2c_sep(re_t, im_t, True)
                logger.error("  - %s: FAILED | wrong-dtype input was accepted", name)
                return False
            except RuntimeError as exc:
                if "float32" not in str(exc):
                    logger.error("  - %s: FAILED | unexpected error: %s", name, exc)
                    return False
            logger.info("  - dtype negative (%s): PASSED", name)
        self.passed_tests += 1
        return True

    def run_test_sep(self, cfg: FftSepConfig):
        """
        执行分离模式 FFT 测试。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据与 CPU 参考值
        in_real, in_imag = self.get_sep_data(cfg.shape, self.device)
        x_cpu = torch.complex(in_real.cpu(), in_imag.cpu())
        dims = tuple(range(-cfg.transform_dims, 0))

        # 逻辑拆分：根据方向计算参考值
        if cfg.is_forward:
            ref = torch.fft.fftn(x_cpu, dim=dims)
        else:
            # 逆变换缩放计算
            scale = 1.0
            for d in dims:
                scale *= x_cpu.size(d)
            ref = torch.fft.ifftn(x_cpu, dim=dims) * scale

        direction_str = "Forward" if cfg.is_forward else "Inverse"
        label = f"{cfg.transform_dims}D {direction_str} FFT (Sep) {cfg.shape}"

        # 2. 调用 NPU 接口 (G.ERR.01: 仅包裹核心算子调用)
        try:
            # 执行前热身
            _ = torch_sip.asd_fft_c2c_sep(in_real, in_imag, cfg.is_forward)
            start_ts = time.time()
            out_real, out_imag = torch_sip.asd_fft_c2c_sep(
                in_real, in_imag, cfg.is_forward
            )
            duration = (time.time() - start_ts) * 1000
        except Exception as exc:
            # 异常处理拆分
            if cfg.expect_fail:
                logger.info(
                    "  ✓ %s: PASSED (Caught expected error: %s...)",
                    label,
                    str(exc)[:50],
                )
                self.passed_tests += 1
                self.total_tests += 1
            else:
                logger.error("  ✗ %s: Unexpected Execution Error -> %s", label, exc)
                self.total_tests += 1
            return

        # 3. 正常执行后的校验逻辑
        if cfg.expect_fail:
            logger.error(
                "  ✗ %s: FAILED (Expected it to fail, but it executed successfully)",
                label,
            )
            self.total_tests += 1
            return

        self._validate_result(label, out_real, out_imag, ref)
        logger.info("    - Execution Time: %.2f ms", duration)

    def _validate_result(
        self,
        label: str,
        out_real: torch.Tensor,
        out_imag: torch.Tensor,
        ref_complex: torch.Tensor,
    ) -> bool:
        """内部校验逻辑：合并复数并比对"""
        self.total_tests += 1
        # 合并 NPU 结果
        npu_complex = torch.complex(out_real.cpu(), out_imag.cpu())
        is_close = torch.allclose(npu_complex, ref_complex, atol=2e-3, rtol=1e-3)

        if is_close:
            logger.info("  ✓ %s: PASSED", label)
            self.passed_tests += 1
            return True

        max_diff = (npu_complex - ref_complex).abs().max()
        logger.error("  ✗ %s: FAILED | Max Diff: %.6e", label, max_diff)
        return False


def main():
    """主测试套件入口"""
    tester = FFTSesterSep()

    # 1. 1D 分离测试
    logger.info("\n" + "=" * 20 + " 1D Sep FFT Scenarios " + "=" * 20)
    tester.run_test_sep(FftSepConfig((16, 128), is_forward=True, name="1D_Fwd"))
    tester.run_test_sep(FftSepConfig((32, 100), is_forward=True, name="1D_Fwd_Non2n"))
    tester.run_test_sep(FftSepConfig((16, 128), is_forward=False, name="1D_Inv"))

    # 2. 2D 分离测试 (预期失败场景)
    logger.info("\n" + "=" * 20 + " 2D Sep FFT (Expect Fail) " + "=" * 20)
    tester.run_test_sep(
        FftSepConfig(
            (8, 4, 1024),
            is_forward=True,
            transform_dims=2,
            expect_fail=True,
            name="2D_Fwd",
        )
    )

    # 3. 3D 分离测试
    logger.info("\n" + "=" * 20 + " 3D Sep FFT Scenarios " + "=" * 20)
    tester.run_test_sep(
        FftSepConfig((2, 16, 32, 64), is_forward=True, transform_dims=3, name="3D_Fwd")
    )
    tester.run_test_sep(
        FftSepConfig((2, 16, 32, 64), is_forward=False, transform_dims=3, name="3D_Inv")
    )

    # 4. 错误 dtype 负向用例（issue #371）
    logger.info("\n" + "=" * 20 + " Dtype Negative " + "=" * 20)
    tester.run_dtype_negative_case()

    # 总结报告
    logger.info("\n" + "=" * 50)
    logger.info(
        "Final Report (Sep-Mode): %d/%d Tests Passed",
        tester.passed_tests,
        tester.total_tests,
    )
    logger.info("=" * 50)
    return 0 if tester.passed_tests == tester.total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
