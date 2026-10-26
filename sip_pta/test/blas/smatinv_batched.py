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

import torch
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_smatinv_test")


@dataclass(frozen=True)
class InvertConfig:
    """参数对象：封装批量实数矩阵求逆的维度、精度及名称配置"""

    batch: int
    n: int
    dtype: torch.dtype = torch.float32
    name: str = "Case"


class SmatinvBatchedTester:
    """批量实数矩阵求逆测试类 (torch_sip.npu_inv, 对齐 torch.linalg.inv)"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info(
            "当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env
        )

    def get_safe_invertible_mat(self, cfg: InvertConfig) -> torch.Tensor:
        """
        生成可逆实数矩阵。
        绝对对角占优构造: 每行对角元取该行其余元素绝对值之和 + 1,
        保证严格对角占优 -> 非奇异 (randn 谱半径 ~2*sqrt(n), 固定偏移
        在大 n 下不充分, n=128 起 randn+5I 可近奇异)。
        """
        shape = (cfg.batch, cfg.n, cfg.n)

        # 1. 在 CPU 上生成数据以规避 NPU 随机数生成限制
        a = torch.randn(shape, dtype=torch.float32)

        # 2. 行内绝对对角占优
        off_diag = a.abs().sum(dim=-1) - a.abs().diagonal(dim1=-2, dim2=-1)
        a = a + torch.diag_embed(off_diag + 1.0)

        return a.to(cfg.dtype).to(self.device)

    def run_case(self, cfg: InvertConfig) -> bool:
        """
        执行单次求逆测试。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 数据准备
        mat_a = self.get_safe_invertible_mat(cfg)
        a_input = mat_a.contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹算子核心逻辑)
        try:
            a_inv_npu = torch_sip.npu_inv(a_input)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (linalg.inv 不支持 Half/BFloat16, 统一 fp32 域)
        ref_inv = torch.linalg.inv(mat_a.cpu().float())

        # 4. 精度校验 (低精度内部以 fp32 计算后回转, 输出同转 fp32 对比)
        res_cpu = a_inv_npu.cpu().float()

        # fp16/bf16 内部以 fp32 计算后回转, 精度放宽
        if cfg.dtype == torch.float32:
            rtol, atol = 1e-4, 1e-4
        else:
            rtol, atol = 1e-2, 1e-2
        is_close = torch.allclose(res_cpu, ref_inv, rtol=rtol, atol=atol)

        status = "PASS" if is_close else "FAIL"
        dtype_str = str(cfg.dtype).replace("torch.", "")
        logger.info(
            "[%s] %-20s | %s | B=%d, N=%d",
            status,
            cfg.name,
            dtype_str,
            cfg.batch,
            cfg.n,
        )

        if not is_close:
            max_diff = torch.abs(res_cpu - ref_inv).max()
            logger.error("      Max Diff: %.6e", max_diff)

        return is_close

    def run_backward_case(self, cfg: InvertConfig) -> bool:
        """
        反向传播测试: 自动建图 (y.backward) 与手动两阶段 (npu_inv_backward) 双路径校验。
        dA = -(Y^T dY) Y^T, 与 torch autograd 解析结果对比。
        """
        mat_a = self.get_safe_invertible_mat(cfg)
        # 低精度输入在 fp32 域校验 (反向参考与梯度对比统一 fp32)
        a32 = mat_a.float().detach()
        dy32 = torch.randn_like(a32)

        # 路径 1: 自动建图
        a1 = a32.clone().requires_grad_(True)
        y1 = torch_sip.npu_inv(a1)
        y1.backward(dy32)
        grad_auto = a1.grad.cpu()

        # 路径 2: 手动两阶段 (no_grad 正向 + 纯反向)
        with torch.no_grad():
            y2 = torch_sip.npu_inv(a32)
        grad_manual = torch_sip.npu_inv_backward(y2, dy32).cpu()

        # 3. CPU 参考: torch.linalg.inv 的 autograd 解析梯度
        a_ref = a32.cpu().requires_grad_(True)
        y_ref = torch.linalg.inv(a_ref)
        y_ref.backward(dy32.cpu())
        grad_ref = a_ref.grad

        ok_auto = torch.allclose(grad_auto, grad_ref, rtol=1e-3, atol=1e-3)
        ok_manual = torch.allclose(grad_manual, grad_ref, rtol=1e-3, atol=1e-3)

        status = "PASS" if (ok_auto and ok_manual) else "FAIL"
        logger.info(
            "[%s] %-20s | backward auto=%s manual=%s | B=%d, N=%d",
            status,
            cfg.name,
            ok_auto,
            ok_manual,
            cfg.batch,
            cfg.n,
        )

        if not (ok_auto and ok_manual):
            logger.error(
                "      Auto Max Diff: %.6e | Manual Max Diff: %.6e",
                torch.abs(grad_auto - grad_ref).max(),
                torch.abs(grad_manual - grad_ref).max(),
            )

        return ok_auto and ok_manual


def main():
    """主测试套件"""
    tester = SmatinvBatchedTester()
    logger.info("开始 SMATINV_BATCHED 实数求逆测试 ...\n")

    # 参数对象化定义: 覆盖小矩阵 (n<32, Gauss-Jordan 分支) 与
    # 32<=n<=256 (SgetriBatched/P2 分支), n%16!=0 走 pad 补齐
    test_suites = [
        InvertConfig(3, 8, name="FP32_Small"),
        InvertConfig(8, 24, name="FP32_N24"),
        InvertConfig(8, 128, name="FP32_P2"),  # n%16==0 直接进 P2
        InvertConfig(8, 96, name="FP32_P2_N96"),  # n%16==8 直接进 P2
        InvertConfig(8, 100, name="FP32_Pad"),  # n%16=4 走 pad 到 104
        InvertConfig(128, 256, name="FP32_Large"),
        InvertConfig(3, 8, dtype=torch.float16, name="FP16_Small"),
        InvertConfig(3, 8, dtype=torch.bfloat16, name="BF16_Small"),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    # 反向双路径专项 (fp32)
    logger.info("")
    bwd_suites = [
        InvertConfig(8, 24, name="BWD_N24"),
        InvertConfig(8, 128, name="BWD_P2"),
    ]
    for config in bwd_suites:
        if not tester.run_backward_case(config):
            all_passed = False

    logger.info("-" * 60)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
