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
import torch_npu
import torch.nn.functional as F
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_convolve")


@dataclass(frozen=True)
class ConvConfig:
    """参数对象：封装卷积测试的维度、类型及名称配置"""

    batch: int
    signal_len: int
    kernel_len: int
    complex_dtype: torch.dtype = torch.complex64
    name: str = "Case"


class ConvolveTester:
    """1D 复数信号卷积算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "Async" if blocking_env == "0" else "Blocking"
        logger.info("当前 NPU 模式: %s (BLOCKING=%s)", mode_str, blocking_env)

    # --- Static Methods ---

    @staticmethod
    def calculate_reference(
        u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_len = v.shape[0]
        u_in = u.unsqueeze(1)
        v_in = v.flip(0).view(1, 1, -1).to(u.dtype)
        pad_l = k_len // 2
        pad_r = k_len - 1 - pad_l
        ref_same = F.conv1d(F.pad(u_in, (pad_l, pad_r)), v_in).squeeze(1)
        ref_causal = F.conv1d(F.pad(u_in, (k_len - 1, 0)), v_in).squeeze(1)
        return ref_same, ref_causal

    @staticmethod
    def debug_log_elements(npu: torch.Tensor, same: torch.Tensor, causal: torch.Tensor):
        def fmt(t):
            return " ".join([f"{x.real:6.2f}+{x.imag:5.2f}j" for x in t[0, :6]])

        logger.info("  [NPU Output ] %s", fmt(npu))
        logger.info("  [Ref (Same) ] %s", fmt(same))
        logger.info("  [Ref (Causal)] %s", fmt(causal))

    @staticmethod
    def log_section(title: str):
        logger.info("\n" + "=" * 20 + f" {title} " + "=" * 20)

    # --- Public Test APIs ---

    def run_case(self, cfg: ConvConfig) -> bool:
        """
        执行单次卷积测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据
        signal = self._get_complex_signal(
            (cfg.batch, cfg.signal_len), cfg.complex_dtype
        )
        kernel = self._get_real_kernel((cfg.kernel_len,), cfg.complex_dtype)

        # 2. 构建 CPU Reference
        sig_ref = signal.cpu().to(torch.complex64)
        ker_ref = kernel.cpu().to(torch.float32)
        ref_same, ref_causal = self.calculate_reference(sig_ref, ker_ref)

        # 3. 核心执行 (G.ERR.01: try 仅包裹核心算子)
        try:
            torch_npu.npu.synchronize()
            npu_out = torch_sip.asd_convolve(signal, kernel)
            torch_npu.npu.synchronize()
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 4. 精度比对与模式识别
        npu_res_c64 = npu_out.cpu().to(torch.complex64)
        rtol, atol = (
            (1e-3, 1e-3) if cfg.complex_dtype == torch.complex64 else (1e-2, 1e-2)
        )

        is_same = torch.allclose(npu_res_c64, ref_same, rtol=rtol, atol=atol)
        is_causal = torch.allclose(npu_res_c64, ref_causal, rtol=rtol, atol=atol)

        if is_same or is_causal:
            mode_tag = "Centered SAME" if is_same else "Causal Filter"
            logger.info(
                "[PASS] %-20s | 模式: %s | B=%d, L=%d",
                cfg.name,
                mode_tag,
                cfg.batch,
                cfg.signal_len,
            )
            return True

        # 失败处理
        logger.error("[FAIL] %-20s | 精度模式不匹配!", cfg.name)
        self.debug_log_elements(npu_res_c64, ref_same, ref_causal)
        return False

    def test_exception_cases(self) -> bool:
        """
        测试适配层参数拦截逻辑。
        G.ERR.01: 使用 try-except-else 结构。
        """
        logger.info("\n" + "=" * 20 + " 开始异常拦截测试 " + "=" * 20)
        all_intercepted = True

        error_suites = [
            (1, 11, 10, "信号过短 (<12)"),
            (1, 30000, 10, "信号过长 (>26208)"),
            (1, 100, 7, "卷积核过短 (<8)"),
            (1, 100, 33, "卷积核过长 (>32)"),
            (800, 100, 10, "Batch过大 (>768)"),
        ]

        for b, s, k, msg in error_suites:
            sig = (
                torch.randn((b, s), dtype=torch.float32)
                .to(self.device)
                .to(torch.complex64)
            )
            ker = torch.randn((k,), dtype=torch.float32).to(self.device)

            try:
                # 核心调用
                torch_sip.asd_convolve(sig, ker)
            except RuntimeError as exc:
                # 预期路径
                logger.info("[PASS] %s 拦截成功: %s...", msg, str(exc)[:50])
            else:
                # 拦截失败路径 (G.ERR.01 规范逻辑)
                logger.error("[FAIL] %s 未被拦截!", msg)
                all_intercepted = False

        # 2 维 kernel 与精度交叉组合负向用例（issue #387）：非 [k] 一维 kernel 与
        # 未配对精度（ComplexFloat↔Float32、ComplexHalf↔Float16）均应在绑定层拦截。
        # kernel 按用例 dtype 直接构造（_get_real_kernel 按 signal dtype 映射精度，
        # 无法表达交叉组合）；并断言异常来自对应绑定层校验（对齐 swap_last2_axes
        # #370 口径，宽泛 except 会把其他来源错误误判为拦截生效）
        error_shape_dtype_suites = [
            (
                (1, 100),
                (2, 16),
                torch.complex64,
                torch.float32,
                "1D tensor",
                "2 维 kernel 静默取 dims[0]",
            ),
            (
                (1, 100),
                (16,),
                torch.complex64,
                torch.float16,
                "must be paired",
                "ComplexFloat signal + Float16 kernel",
            ),
            (
                (1, 100),
                (16,),
                torch.complex32,
                torch.float32,
                "must be paired",
                "ComplexHalf signal + Float32 kernel",
            ),
        ]

        for (
            sig_shape,
            ker_shape,
            sig_dtype,
            ker_dtype,
            keyword,
            msg,
        ) in error_shape_dtype_suites:
            sig = self._get_complex_signal(sig_shape, sig_dtype)
            ker = torch.randn(ker_shape, dtype=ker_dtype).to(self.device)

            try:
                torch_sip.asd_convolve(sig, ker)
            except RuntimeError as exc:
                if keyword not in str(exc):
                    logger.error(
                        "[FAIL] %s | unexpected error: %s...", msg, str(exc)[:80]
                    )
                    all_intercepted = False
                else:
                    logger.info("[PASS] %s 拦截成功: %s...", msg, str(exc)[:50])
            else:
                logger.error("[FAIL] %s 未被拦截!", msg)
                all_intercepted = False

        return all_intercepted

    # --- Internal Helpers ---

    def _get_complex_signal(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        f_dtype = torch.float16 if dtype == torch.complex32 else torch.float32
        real = torch.randn(shape, dtype=f_dtype)
        imag = torch.randn(shape, dtype=f_dtype)
        return torch.complex(real, imag).to(self.device)

    def _get_real_kernel(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        f_dtype = torch.float16 if dtype == torch.complex32 else torch.float32
        return torch.randn(shape, dtype=f_dtype).to(self.device)


def main() -> int:
    """测试主入口，返回退出码"""
    tester = ConvolveTester()
    all_passed = True

    # 极值
    max_sig_len = 26208
    max_batch = 768

    test_suites = [
        ConvConfig(2, 128, 16, name="Normal_C64_Even"),
        ConvConfig(8, 256, 31, name="Normal_C64_Odd"),
        ConvConfig(1, 12, 8, name="Min_Bound_C64"),
        ConvConfig(max_batch, max_sig_len, 32, name="Max_Bound_C64"),
        ConvConfig(2, 128, 16, torch.complex32, "Normal_C32"),
    ]

    ConvolveTester.log_section("常规与连通性测试")
    for cfg in test_suites:
        if not tester.run_case(cfg):
            all_passed = False

    # 异常拦截测试汇总
    if not tester.test_exception_cases():
        all_passed = False

    logger.info("-" * 50)
    if all_passed:
        logger.info("测试报告: ✅ 所有项通过")
        return 0
    else:
        logger.error("测试报告: ❌ 存在失败项")
        return 1


if __name__ == "__main__":
    sys.exit(main())
