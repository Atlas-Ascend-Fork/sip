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
import torch_npu  # noqa: F401  # torch.npu 属性由 torch_npu 导入时注册，不可移除

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_test")


def get_torch_sip():
    """
    检查并导入 torch_sip 模块。
    """
    try:
        import torch_sip

        logger.info("✓ torch_sip module loaded successfully.")
        return torch_sip
    except ImportError as e:
        error_msg = (
            f"Failed to import torch_sip: {e}\n"
            "Please build extension first:\n"
            "  cd torch_sip && python setup.py build_ext --inplace"
        )
        raise ImportError(error_msg) from e


def initialize_npu():
    """
    初始化 NPU 设备。
    """
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available. This test requires NPU hardware.")

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    logger.info("✓ NPU initialized. Using device: %s", device)
    return device


def verify_complex_mul(result, expected):
    """
    验证复数乘法结果。
    """
    res_cpu = result.cpu()
    exp_cpu = expected.cpu()

    if torch.allclose(res_cpu, exp_cpu, rtol=1e-3, atol=1e-3):
        logger.info("✓ Precision check passed: result matches expected (x * y).")
        return True

    logger.error("✗ Precision check failed!")
    logger.debug("Result sample: %s", res_cpu.flatten()[:2])
    logger.debug("Expect sample: %s", exp_cpu.flatten()[:2])
    return False


def test_asd_mul(torch_sip, device):
    """
    执行 asd_mul 测试逻辑。
    """
    logger.info("-" * 40)
    logger.info("Testing asd_mul operation...")

    # 构造输入数据
    shape = (4, 4)
    x = torch.complex(
        torch.randn(shape, device=device), torch.randn(shape, device=device)
    )
    y = torch.complex(
        torch.randn(shape, device=device), torch.randn(shape, device=device)
    )

    logger.info("Input tensors: shape=%s, dtype=%s", x.shape, x.dtype)

    # 计算预期值与实际值
    expected = x * y
    result = torch_sip.asd_mul(x, y)

    return verify_complex_mul(result, expected)


def test_asd_mul_negative(torch_sip, device):
    """
    一致性校验负向用例（issue #384）：形状/dtype/设备不一致的输入应在绑定层
    报参数错误，不再下传内核（y 元素数不足时存在设备侧越界读风险）。
    """
    logger.info("-" * 40)
    logger.info("Testing asd_mul negative cases (issue #384)...")

    all_intercepted = True
    shape = (4, 4)
    x = torch.complex(
        torch.randn(shape, device=device), torch.randn(shape, device=device)
    )

    negative_suites = [
        (
            "shape mismatch (y [2,4] vs x [4,4])",
            "must have the same shape",
            torch.complex(
                torch.randn((2, 4), device=device), torch.randn((2, 4), device=device)
            ),
        ),
        (
            "dtype mismatch (y float32 vs x complex64)",
            "must have the same dtype",
            torch.randn(shape, device=device),
        ),
        (
            "device mismatch (y on CPU vs x on NPU)",
            "must be on the same device",
            torch.complex(torch.randn(shape), torch.randn(shape)),
        ),
    ]

    for msg, keyword, y in negative_suites:
        try:
            torch_sip.asd_mul(x, y)
        except RuntimeError as exc:
            # 断言异常来自新增绑定层校验（对齐 swap_last2_axes #370 口径）：
            # 宽泛 except 会把内核/aclnn 侧的设备错误误判为校验生效
            if keyword not in str(exc):
                logger.error(
                    "  [FAIL] %s | unexpected error: %s...", msg, str(exc)[:80]
                )
                all_intercepted = False
            else:
                logger.info("  [PASS] %s 拦截成功: %s...", msg, str(exc)[:60])
        else:
            logger.error("  [FAIL] %s 未被拦截!", msg)
            all_intercepted = False

    return all_intercepted


def main():
    """
    主入口函数。负责异常捕获和最终退出码返回。
    """
    logger.info("Starting torch_sip extension test...")
    logger.info("PyTorch: %s | Python: %s", torch.__version__, sys.version.split()[0])

    try:
        # 1. 环境与设备准备
        sip_module = get_torch_sip()
        npu_device = initialize_npu()

        # 2. 运行测试用例
        success = test_asd_mul(sip_module, npu_device)
        success = test_asd_mul_negative(sip_module, npu_device) and success

        # 3. 汇总结果
        logger.info("=" * 40)
        if success:
            logger.info("Status: ALL TESTS PASSED")
            return 0

        logger.error("Status: TEST FAILED")
        return 1

    except (ImportError, RuntimeError) as e:
        logger.error("Environment Error: %s", e)
        return 1
    except Exception:
        logger.exception("An unexpected fatal error occurred during testing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
