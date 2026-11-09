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
import traceback

import torch
import torch_npu  # noqa: F401  (side-effect: registers the npu device)
import torch_sip

# 配置全局日志 (G.LOG.02 规范)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_test")


def run_test_case(shape, name="Test Case"):
    """通用的测试逻辑"""
    logger.info("\n--- %s: Shape %s ---", name, shape)

    if not torch.npu.is_available():
        logger.error("✗ NPU is not available.")
        return False

    # 1. 创建数据
    x = torch.randn(*shape, dtype=torch.complex64).npu()
    logger.info(
        "Input: shape=%s, dtype=%s, is_contiguous=%s",
        x.shape,
        x.dtype,
        x.is_contiguous(),
    )

    # 2. 预期结果 (使用 PyTorch 自带的 transpose)
    expected = x.transpose(-1, -2)

    try:
        # 3. 调用C++ 扩展算子
        result = torch_sip.swap_last2_axes(x)

        logger.info("✓ Execution successful. Result shape: %s", result.shape)

        # 4. 验证维度
        if result.shape != expected.shape:
            logger.error(
                "✗ Shape mismatch: got %s, expect %s", result.shape, expected.shape
            )
            return False

        # 5. 数值比对 (转回 CPU 比较)
        result_cpu = result.cpu()
        expected_cpu = expected.cpu()

        if torch.allclose(result_cpu, expected_cpu, rtol=1e-4, atol=1e-4):
            logger.info("✓ Numerical result matches PyTorch transpose!")

            # 6. 安全打印第一个元素
            idx = tuple([0] * result_cpu.dim())
            logger.info("  Sample [at %s]:", idx)
            logger.info("    Expected: %s", expected_cpu[idx])
            logger.info("    Got:      %s", result_cpu[idx])
            return True
        else:
            logger.error("✗ Numerical result does not match!")
            return False

    except Exception as exc:
        logger.error("✗ Test failed with error: %s", exc)
        logger.error(traceback.format_exc())
        return False


def test_boundary_cases():
    """测试边界情况"""
    logger.info("\n--- Testing Boundary Case: 1D Tensor ---")
    try:
        x = torch.randn(5).npu()
        # API 名为 swap_last2_axes（torch_sip/__init__.py 与 C++ 注册名一致）；
        # 原用例误写 swap_last_2_axes 抛 AttributeError 被宽泛 except 吞掉，
        # 1D 负向校验从未真实执行（issue #370）
        torch_sip.swap_last2_axes(x)
        logger.error("✗ Failed: Should have raised error for 1D input.")
        return False
    except RuntimeError as exc:
        # 算子侧 1D 维度校验以 RuntimeError（TORCH_CHECK）抛出；断言收紧为校验
        # 异常消息含维度提示，避免 API 名错误（AttributeError）等无关异常被
        # 误判为通过
        if "at least 2 dimensions" not in str(exc):
            logger.error("✗ Unexpected error content: %s", exc)
            return False
        logger.info("✓ Caught expected error: %s", exc)
        return True


def main():
    logger.info("=" * 60)
    logger.info("torch_sip Extension - Swap Operator Comprehensive Test")
    logger.info("PyTorch Version: %s", torch.__version__)
    logger.info("=" * 60)

    success = True

    # Case 1: 2D Tensor
    success &= run_test_case((3, 2), "2D Complex Tensor")
    # Case 2: 3D Tensor (验证storageDims 兼容性)
    success &= run_test_case((2, 4, 5), "3D Complex Tensor (Batch)")
    # Case 3: 异常处理测试
    success &= test_boundary_cases()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("OVERALL STATUS: ALL TESTS PASSED! ✓")
    else:
        logger.error("OVERALL STATUS: SOME TESTS FAILED. ✗")
    logger.info("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
