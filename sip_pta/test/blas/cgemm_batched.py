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
from typing import Tuple, Optional, Dict, Any

import torch
import torch_npu  # noqa: F401  (side-effect: registers the npu device)
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_batched_gemm")


@dataclass(frozen=True)
class BatchedGemmConfig:
    """参数对象：封装 Batched GEMM 的所有维度、模式及类型配置"""

    batch: int
    m: int
    n: int
    k: int
    alpha: Optional[float] = 1.0
    beta: Optional[float] = 0.0
    trans_a: str = "N"
    trans_b: str = "N"
    dtype: torch.dtype = torch.complex64
    name: str = "Case"

    def get_kwargs(self) -> Dict[str, Any]:
        """提取算子所需的关键字参数"""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
        }


class CgemmBatchedTester:
    """Batched GEMM 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info(
            "当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env
        )

    @staticmethod
    def log_header(title: str):
        """静态工具方法：打印测试标题"""
        logger.info("\n%s %s %s", "=" * 20, title, "=" * 20)

    def get_complex_mat_3d(
        self, shape: Tuple[int, int, int], dtype: torch.dtype
    ) -> torch.Tensor:
        """安全生成复数矩阵 (先 CPU 后搬移，规避 NPU 随机数生成限制)"""
        f_dtype = torch.float16 if dtype == torch.complex32 else torch.float32
        real = torch.randn(shape, dtype=f_dtype)
        imag = torch.randn(shape, dtype=f_dtype)
        return torch.complex(real, imag).to(self.device).contiguous()

    def run_case(self, cfg: BatchedGemmConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备输入张量
        mat_a = self.get_complex_mat_3d((cfg.batch, cfg.m, cfg.k), cfg.dtype)
        mat_b = self.get_complex_mat_3d((cfg.batch, cfg.k, cfg.n), cfg.dtype)
        mat_c = self.get_complex_mat_3d((cfg.batch, cfg.m, cfg.n), cfg.dtype)

        # 2. 备份初始值用于 CPU 参考计算 (处理 Complex32 的特殊视图转换)
        if cfg.dtype == torch.complex32:
            # Complex32 搬运需先转回 float 视图
            a_cpu = (
                mat_a.view(torch.float32)
                .cpu()
                .view(torch.complex32)
                .to(torch.complex64)
            )
            b_cpu = (
                mat_b.view(torch.float32)
                .cpu()
                .view(torch.complex32)
                .to(torch.complex64)
            )
            c_init_cpu = (
                mat_c.view(torch.float32)
                .cpu()
                .view(torch.complex32)
                .to(torch.complex64)
            )
        else:
            a_cpu = mat_a.cpu()
            b_cpu = mat_b.cpu()
            c_init_cpu = mat_c.clone().cpu()

        # 3. NPU 计算 (G.ERR.01: 仅包裹核心算子执行)
        try:
            torch_sip.asd_blas_cgemm_batched(mat_a, mat_b, mat_c, **cfg.get_kwargs())
        except Exception as exc:
            logger.error("[%s] 运行时崩溃: %s", cfg.name, exc)
            return False

        # 4. CPU 参考结果计算 (处理转置逻辑)
        def apply_trans(m, op):
            op = op.upper()
            return (
                m.transpose(1, 2)
                if op == "T"
                else (m.conj().transpose(1, 2) if op == "C" else m)
            )

        ref_a = apply_trans(a_cpu, cfg.trans_a)
        ref_b = apply_trans(b_cpu, cfg.trans_b)
        ref = cfg.alpha * torch.bmm(ref_a, ref_b) + cfg.beta * c_init_cpu

        # 5. 精度校验
        rtol, atol = (1e-3, 1e-3) if cfg.dtype == torch.complex32 else (1e-4, 1e-4)
        res_npu_cpu = mat_c.cpu().to(torch.complex64)
        is_close = torch.allclose(res_npu_cpu, ref, rtol=rtol, atol=atol)

        status = "PASS" if is_close else "FAIL"
        dtype_tag = "C32" if cfg.dtype == torch.complex32 else "C64"
        logger.info(
            "[%s] %-25s | %s | B=%d, M=%d, N=%d, K=%d",
            status,
            cfg.name,
            dtype_tag,
            cfg.batch,
            cfg.m,
            cfg.n,
            cfg.k,
        )

        if not is_close:
            logger.error("      Max Diff: %.6f", (res_npu_cpu - ref).abs().max())

        return is_close


def main():
    """主测试套件入口"""
    tester = CgemmBatchedTester()
    CgemmBatchedTester.log_header("CGEMM Batched 专项测试启动")

    # 定义测试套件 (使用参数对象模式)
    test_suites = [
        BatchedGemmConfig(2, 4, 4, 4, name="FP32_Default"),
        BatchedGemmConfig(1, 8, 8, 8, alpha=1.0, beta=0.0, name="FP32_Alpha_Beta"),
        BatchedGemmConfig(2, 8, 16, 24, dtype=torch.complex32, name="FP16_Rect_1"),
        BatchedGemmConfig(4, 32, 8, 16, dtype=torch.complex32, name="FP16_Rect_2"),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    # 连续切片视图（storage_offset != 0）输入用例（issue #368）：
    # 修复前 CreateAclTensorFromAtTensor 以 storage 首地址为基址，切片视图读写整体
    # 错位；修复后基址取 data_ptr()，切片视图结果应与整块物化后切片一致（bit 级）
    logger.info("-" * 50)
    logger.info("Contiguous slice-view case (issue #368):")
    # 张量构造对齐本文件 get_complex_mat_3d 的 float 实部/虚部模式：
    # torch.randn 直接以 complex64 在 NPU 上创建时 aclnnInplaceNormal 不支持
    # DT_COMPLEX64（torch_npu 2.10 实测报 161002），complex 张量经 CPU 构造后搬运
    # trans 入参对齐包装层字符串契约（int 会先在 _get_trans_enum 处 AttributeError）
    big_a = torch.complex(
        torch.randn(6, 8, 16, dtype=torch.float32),
        torch.randn(6, 8, 16, dtype=torch.float32),
    ).to("npu")
    big_b = torch.complex(
        torch.randn(6, 16, 24, dtype=torch.float32),
        torch.randn(6, 16, 24, dtype=torch.float32),
    ).to("npu")
    big_c = torch.complex(
        torch.randn(6, 8, 24, dtype=torch.float32),
        torch.randn(6, 8, 24, dtype=torch.float32),
    ).to("npu")
    a_view = big_a[2:]  # [4, 8, 16] 连续视图，storage_offset != 0
    b_view = big_b[2:]
    c_view = big_c[2:].clone()
    torch_sip.asd_blas_cgemm_batched(
        a_view, b_view, c_view, alpha=1.0, beta=0.0, trans_a="N", trans_b="N"
    )
    c_ref = big_c[2:].clone()
    torch_sip.asd_blas_cgemm_batched(
        a_view.contiguous().clone(),
        b_view.contiguous().clone(),
        c_ref,
        alpha=1.0,
        beta=0.0,
        trans_a="N",
        trans_b="N",
    )
    if torch.equal(c_view.cpu(), c_ref.cpu()):
        logger.info("  - slice-view cgemm_batched: PASSED (bit-exact)")
    else:
        logger.error(
            "  - slice-view cgemm_batched: FAILED | slice view result mismatch"
        )
        all_passed = False

    # 形状一致性负向用例（issue #386）：批数不一致与 k/m/n 不匹配均应在绑定层
    # 报参数错误，不再按 C 侧 batchCount 静默下传（batch 不一致时存在越界读风险）。
    # 包装层 trans 入参为字符串，合法 "N" 经包装层下传触达 C++ 形状校验
    logger.info("-" * 50)
    logger.info("Shape negative cases (issue #386):")
    negative_suites = [
        (
            "batch count mismatch (A/B [4] vs C [2])",
            (4, 8, 16),
            (4, 16, 24),
            (2, 8, 24),
        ),
        (
            "k mismatch (A.size(2)=16 vs B.size(1)=12)",
            (2, 8, 16),
            (2, 12, 24),
            (2, 8, 24),
        ),
        (
            "n mismatch (B.size(2)=24 vs C.size(2)=16)",
            (2, 8, 16),
            (2, 16, 24),
            (2, 8, 16),
        ),
        (
            "m mismatch (A.size(1)=8 vs C.size(1)=4)",
            (2, 8, 16),
            (2, 16, 24),
            (2, 4, 24),
        ),
    ]

    for msg, a_shape, b_shape, c_shape in negative_suites:
        # complex 张量经 float 实部/虚部构造后搬运：aclnnInplaceNormal 不支持
        # DT_COMPLEX64（torch_npu 2.10 实测 161002），见上方切片用例同款注释
        a_neg = torch.complex(
            torch.randn(a_shape, dtype=torch.float32),
            torch.randn(a_shape, dtype=torch.float32),
        ).to("npu")
        b_neg = torch.complex(
            torch.randn(b_shape, dtype=torch.float32),
            torch.randn(b_shape, dtype=torch.float32),
        ).to("npu")
        c_neg = torch.complex(
            torch.randn(c_shape, dtype=torch.float32),
            torch.randn(c_shape, dtype=torch.float32),
        ).to("npu")
        try:
            torch_sip.asd_blas_cgemm_batched(
                a_neg, b_neg, c_neg, alpha=1.0, beta=0.0, trans_a="N", trans_b="N"
            )
        except RuntimeError as exc:
            logger.info("  [PASS] %s 拦截成功: %s...", msg, str(exc)[:60])
        else:
            logger.error("  [FAIL] %s 未被拦截!", msg)
            all_passed = False

    logger.info("-" * 50)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
