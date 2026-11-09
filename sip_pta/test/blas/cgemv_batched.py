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
from typing import Tuple, Union

import torch
import torch_npu  # noqa: F401  (side-effect: registers the npu device)
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_cgemv_batched")


@dataclass(frozen=True)
class BatchedGemvConfig:
    """参数对象：封装 Batched CGEMV 的所有维度、精度及模式配置"""

    batch: int
    m: int
    n: int
    alpha: Union[complex, float]
    beta: Union[complex, float]
    trans: str = "N"
    dtype: torch.dtype = torch.complex64
    name: str = "Case"


class CgemvBatchedTester:
    """Batched CGEMV 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info(
            "当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env
        )

    @staticmethod
    def _prepare_cpu_refs(
        mat_a: torch.Tensor,
        vec_x: torch.Tensor,
        vec_y_init: torch.Tensor,
        cfg: BatchedGemvConfig,
    ):
        """
        静态私有方法：处理 Complex32 到 Complex64 的 CPU 视图转换逻辑。
        整改：由于不依赖实例状态，声明为 @staticmethod。
        """
        if cfg.dtype == torch.complex32:
            # Complex32 在内存上等同于 32bit 的数据（如 float32/int32）
            a_ref = (
                mat_a.view(torch.float32)
                .cpu()
                .view(torch.complex32)
                .to(torch.complex64)
            )
            x_ref = (
                vec_x.view(torch.float32)
                .cpu()
                .view(torch.complex32)
                .to(torch.complex64)
            )
            y_ref = (
                vec_y_init.view(torch.float32)
                .cpu()
                .view(torch.complex32)
                .to(torch.complex64)
            )
        else:
            a_ref = mat_a.cpu()
            x_ref = vec_x.cpu()
            y_ref = vec_y_init.cpu()

        return a_ref, x_ref.unsqueeze(-1), y_ref.unsqueeze(-1)

    def get_complex_tensor(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """安全生成复数张量 (先 CPU 后搬移，规避 NPU 随机数生成限制)"""
        f_dtype = torch.float16 if dtype == torch.complex32 else torch.float32
        real = torch.randn(shape, dtype=f_dtype)
        imag = torch.randn(shape, dtype=f_dtype)
        return torch.complex(real, imag).to(self.device)

    def run_case(self, cfg: BatchedGemvConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 数据准备
        x_len, y_len = cfg.n, cfg.m
        mat_a = self.get_complex_tensor((cfg.batch, cfg.m, cfg.n), dtype=cfg.dtype)
        vec_x = self.get_complex_tensor((cfg.batch, x_len), dtype=cfg.dtype)
        vec_y = self.get_complex_tensor((cfg.batch, y_len), dtype=cfg.dtype)

        # 处理 Complex32 的克隆避坑 (view as int32 确保 bit-wise copy)
        if cfg.dtype == torch.complex32:
            vec_y_init = vec_y.view(torch.int32).clone().view(torch.complex32)
        else:
            vec_y_init = vec_y.clone()

        # 2. 核心调用 (G.ERR.01: 仅包裹算子执行逻辑)
        try:
            torch_sip.asd_blas_cgemv_batched(
                mat_a, vec_x, vec_y, cfg.alpha, cfg.beta, cfg.trans
            )
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        a_ref, x_ref, y_ref = CgemvBatchedTester._prepare_cpu_refs(
            mat_a, vec_x, vec_y_init, cfg
        )

        trans_op = cfg.trans.upper()
        if trans_op == "T":
            a_ref = a_ref.transpose(1, 2)
        elif trans_op == "C":
            a_ref = a_ref.conj().transpose(1, 2)

        ref = torch.baddbmm(
            y_ref, a_ref, x_ref, alpha=cfg.alpha, beta=cfg.beta
        ).squeeze(-1)

        # 4. 精度校验与日志
        rtol, atol = (1e-4, 1e-4) if cfg.dtype == torch.complex64 else (1e-2, 1e-2)
        res_npu_cpu = vec_y.cpu().to(torch.complex64)
        is_close = torch.allclose(res_npu_cpu, ref, rtol=rtol, atol=atol)

        status = "PASS" if is_close else "FAIL"
        dtype_str = "C64" if cfg.dtype == torch.complex64 else "C32"
        logger.info(
            "[%s] %-20s | %s | B=%d, M=%d, N=%d | Trans=%s",
            status,
            cfg.name,
            dtype_str,
            cfg.batch,
            cfg.m,
            cfg.n,
            cfg.trans,
        )

        if not is_close:
            logger.error("      Max Diff: %.6e", (res_npu_cpu - ref).abs().max())

        return is_close


def main():
    """主测试套件入口"""
    tester = CgemvBatchedTester()

    # 使用参数对象定义测试套件 (解决参数过多导致的阅读困难)
    test_suites = [
        # FP32 复数测试
        BatchedGemvConfig(3, 16, 8, 1.0, 0.0, "N", name="FP32_Normal"),
        BatchedGemvConfig(2, 32, 32, 1.0, 0.0, "C", name="FP32_Conj"),
        BatchedGemvConfig(2, 32, 32, 1.0, 0.0, "T", name="FP32_Transpose"),
        # FP16 复数测试 (Complex32)
        BatchedGemvConfig(
            3, 16, 8, 1.0, 0.0, "N", dtype=torch.complex32, name="FP16_Normal"
        ),
        BatchedGemvConfig(
            4, 32, 32, 1.0, 0.0, "T", dtype=torch.complex32, name="FP16_Transpose"
        ),
        BatchedGemvConfig(
            2, 32, 32, 1.0, 0.0, "C", dtype=torch.complex32, name="FP16_Conj"
        ),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    # 连续切片视图（storage_offset != 0）输入用例（issue #368）：
    # 修复前 CreateAclTensorFromAtTensor 以 storage 首地址为基址，切片视图读写整体
    # 错位；修复后基址取 data_ptr()，切片视图结果应与整块物化后切片一致（bit 级）
    logger.info("-" * 60)
    logger.info("Contiguous slice-view case (issue #368):")
    big_a = torch.randn(6, 16, 8, dtype=torch.complex64, device="npu")
    big_x = torch.randn(6, 8, dtype=torch.complex64, device="npu")
    big_y = torch.randn(6, 16, dtype=torch.complex64, device="npu")
    a_view = big_a[2:]  # [4, 16, 8] 连续视图，storage_offset != 0
    x_view = big_x[2:]  # [4, 8] 连续视图
    y_view = big_y[2:].clone()  # 输出用独立缓冲
    torch_sip.asd_blas_cgemv_batched(a_view, x_view, y_view, 1.0, 0.0, "N")
    y_ref = big_y[2:].clone()
    torch_sip.asd_blas_cgemv_batched(
        a_view.contiguous().clone(), x_view.contiguous().clone(), y_ref, 1.0, 0.0, "N"
    )
    if torch.equal(y_view.cpu(), y_ref.cpu()):
        logger.info("  - slice-view cgemv_batched: PASSED (bit-exact)")
    else:
        logger.error(
            "  - slice-view cgemv_batched: FAILED | slice view result mismatch"
        )
        all_passed = False

    logger.info("-" * 60)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
