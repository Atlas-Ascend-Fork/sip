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

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_test")


def get_torch_sip():
    try:
        import torch_sip

        logger.info("torch_sip module loaded successfully")
        return torch_sip
    except ImportError as e:
        raise ImportError(
            f"Failed to import torch_sip: {e}\n"
            "Please build extension first:\n"
            "  cd sip_pta && bash build.sh --install"
        ) from e


def _rel_stats(actual, ref):
    """相对误差统计: 分母加 3e-3·rms 地板, 避免近零元素放大 maxRel(与 test_suite/UT 口径一致)。"""
    rms = ref.pow(2).mean().sqrt()
    rel = (actual - ref).abs() / ref.abs().clamp_min(3e-3 * rms)
    return rel.max().item(), rel.mean().item()


def run_case(torch_sip, batch, n, nrhs, seed=42):
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    torch.manual_seed(seed)
    A = torch.randn(batch, n, n, dtype=torch.float32)
    A.diagonal(dim1=-2, dim2=-1).add_(3.0)  # 对角增强保证可逆
    B = torch.randn(batch, n, nrhs, dtype=torch.float32)

    A_npu = A.to(device)
    B_npu = B.to(device)

    X_sip = torch_sip.solve(A_npu, B_npu)
    X_ref = torch.linalg.solve(A, B)

    X_sip_cpu = X_sip.cpu()
    max_rel, mean_rel = _rel_stats(X_sip_cpu, X_ref)

    passed = max_rel < 0.1 and mean_rel < 1e-3
    status = "PASS" if passed else "FAIL"
    logger.info(
        "[%s] batch=%d n=%d nrhs=%d: maxRel=%.6g meanRel=%.6g",
        status,
        batch,
        n,
        nrhs,
        max_rel,
        mean_rel,
    )
    return passed


def run_backward_case(torch_sip, batch, n, nrhs, seed=42):
    """反向对拍: 自动微分路径与手动 solve_backward 路径, 均与 fp32 解析参考比对。"""
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    torch.manual_seed(seed)
    A = torch.randn(batch, n, n, dtype=torch.float32)
    A.diagonal(dim1=-2, dim2=-1).add_(3.0)  # 对角增强保证可逆
    B = torch.randn(batch, n, nrhs, dtype=torch.float32)
    gX = torch.randn(batch, n, nrhs, dtype=torch.float32)

    A_npu = A.to(device)
    B_npu = B.to(device)
    gX_npu = gX.to(device)

    # fp32 解析参考: X = A^-1 B, gB = A^-T gX, gA = -gB X^T
    X_ref = torch.linalg.solve(A, B)
    gB_ref = torch.linalg.solve(A.transpose(-2, -1), gX)
    gA_ref = -(gB_ref @ X_ref.transpose(-2, -1))

    # 路径①: 自动微分 (torch_sip.solve -> _SolveFn -> solve_backward)
    A_g = A_npu.clone().requires_grad_(True)
    B_g = B_npu.clone().requires_grad_(True)
    torch_sip.solve(A_g, B_g).backward(gX_npu)
    gA_auto, gB_auto = A_g.grad.cpu(), B_g.grad.cpu()

    # 路径②: 手动两阶段
    X_man = torch_sip.solve(A_npu, B_npu)
    gB_man, gA_man = torch_sip.solve_backward(A_npu, X_man, gX_npu)

    passed = True
    for tag, gA_val, gB_val in (
        ("autograd", gA_auto, gB_auto),
        ("manual", gA_man.cpu(), gB_man.cpu()),
    ):
        for name, val, ref in (("gA", gA_val, gA_ref), ("gB", gB_val, gB_ref)):
            max_rel, mean_rel = _rel_stats(val, ref)
            ok = max_rel < 0.1 and mean_rel < 1e-3
            passed = passed and ok
            logger.info(
                "[%s] bwd-%s-%s batch=%d n=%d nrhs=%d: maxRel=%.6g meanRel=%.6g",
                "PASS" if ok else "FAIL",
                tag,
                name,
                batch,
                n,
                nrhs,
                max_rel,
                mean_rel,
            )
    return passed


def main():
    torch_sip = get_torch_sip()
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    cases = [
        (4, 2, 1),  # n=2 (N2 路径)
        (4, 2, 4),  # n=2 多列
        (16, 3, 3),  # 小 n
        (8, 16, 4),  # 中 n
        (8, 32, 2),
        (4, 64, 4),  # n=64 整装路径
        (4, 65, 4),  # n=65 大路径(1 panel + 17)
        (4, 96, 8),  # 大路径
        (2, 128, 4),
        (2, 256, 4),  # 大路径上限
        (4, 256, 300),  # 大路径 + 多列块
    ]

    results = []
    for batch, n, nrhs in cases:
        results.append(run_case(torch_sip, batch, n, nrhs))

    # 反向对拍: 自动微分 + 手动路径
    for batch, n, nrhs in [(4, 3, 4), (16, 5, 8), (4, 64, 4)]:
        results.append(run_backward_case(torch_sip, batch, n, nrhs))

    total = len(results)
    passed = sum(results)
    logger.info("=== %d/%d cases passed ===", passed, total)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
