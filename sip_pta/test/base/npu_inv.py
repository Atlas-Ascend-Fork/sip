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

# npu_inv out= 参数校验负向用例（issue #425）：out 的 shape/dtype/device 不匹配时
# 抛出与 torch.linalg.inv 语义一致的错误，autograd 场景（grad 开启且输入 requires_grad）
# 的 out= 显式报错；合法 out 正常拷贝返回。
#
# 说明：本用例通过文件路径直接加载 torch_sip/_npu_inv_impl.py，并以桩替换
# torch.ops.torch_sip.* 自定义算子（返回单位阵），仅验证入口包装逻辑，
# 不依赖编译产物（无需先 build sip_pta）；设备类用例在 torch_npu 可用时执行。

import importlib.util
import logging
import os
import sys
import types
from unittest import mock

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_test")

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMPL_PATH = os.path.join(_HERE, "..", "..", "torch_sip", "_npu_inv_impl.py")


def _load_impl():
    spec = importlib.util.spec_from_file_location("_npu_inv_impl_ut", _IMPL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_eye_stub():
    """torch.ops 桩: 对 [B, N, N] 输入返回单位阵, 仅用于驱动入口包装逻辑"""

    def stub(x):
        b, n, _ = x.shape
        eye = torch.eye(n, dtype=x.dtype, device=x.device)
        return eye.unsqueeze(0).expand(b, n, n).contiguous()

    return stub


def _stub_namespace():
    return types.SimpleNamespace(
        smatinv_inv_forward=_make_eye_stub(),
        smatinv_inv_autograd=_make_eye_stub(),
    )


def _expect_runtime_error(results, name, fn, keyword):
    try:
        fn()
    except RuntimeError as e:
        msg = str(e)
        if keyword in msg:
            results.append(True)
            logger.info("[PASS] %s (msg: %s)", name, msg)
        else:
            results.append(False)
            logger.error("[FAIL] %s: 错误信息未含关键词 %r: %s", name, keyword, msg)
        return
    except Exception as e:  # noqa: BLE001
        results.append(False)
        logger.error("[FAIL] %s: 异常类型不符: %r", name, e)
        return
    results.append(False)
    logger.error("[FAIL] %s: 未抛出 RuntimeError", name)


def run_cases(impl):
    results = []
    ns = _stub_namespace()
    with mock.patch.object(torch.ops, "torch_sip", ns):
        npu_inv = impl.npu_inv
        a = torch.randn(2, 3, 3)

        # 1. 合法 out: 拷贝并返回 out 本体
        out = torch.zeros(2, 3, 3)
        y = npu_inv(a, out=out)
        ok = y is out and torch.equal(out, torch.eye(3).unsqueeze(0).expand(2, 3, 3))
        results.append(ok)
        (logger.info if ok else logger.error)(
            "[PASS] " if ok else "[FAIL] " + "valid out copy returns out itself"
        )

        # 2. shape 不匹配
        _expect_runtime_error(
            results,
            "out shape mismatch",
            lambda: npu_inv(a, out=torch.zeros(3, 3)),
            "shape",
        )

        # 3. dtype 不匹配
        _expect_runtime_error(
            results,
            "out dtype mismatch",
            lambda: npu_inv(a, out=torch.zeros(2, 3, 3, dtype=torch.float16)),
            "dtype",
        )

        # 4. grad 开启且输入 requires_grad 的 out= 调用: 对齐 torch 显式报错
        a_rg = torch.randn(2, 3, 3, requires_grad=True)
        _expect_runtime_error(
            results,
            "autograd with out= rejected",
            lambda: npu_inv(a_rg, out=torch.zeros(2, 3, 3)),
            "automatic differentiation",
        )

        # 5. no_grad 下 requires_grad 输入 + 合法 out: 放行（不过度拒绝）
        out_ng = torch.zeros(2, 3, 3)
        with torch.no_grad():
            y_ng = npu_inv(a_rg, out=out_ng)
        ok = y_ng is out_ng and torch.equal(
            out_ng, torch.eye(3).unsqueeze(0).expand(2, 3, 3)
        )
        results.append(ok)
        (logger.info if ok else logger.error)(
            "[PASS] " if ok else "[FAIL] " + "no_grad + requires_grad + out allowed"
        )

        # 6. grad 开启、输入不 requires_grad、out 为 requires_grad 叶子张量:
        #    对齐 torch 的 out 变体 autograd 守卫（compute_requires_grad(self, out) 口径）
        a_ng = torch.randn(2, 3, 3)
        _expect_runtime_error(
            results,
            "out requires_grad leaf rejected",
            lambda: npu_inv(a_ng, out=torch.zeros(2, 3, 3, requires_grad=True)),
            "automatic differentiation",
        )

        # 7. no_grad 下 out 为 requires_grad 张量: 放行（对齐 torch, 不过度拒绝）
        out_rg = torch.zeros(2, 3, 3, requires_grad=True)
        with torch.no_grad():
            y_rg = npu_inv(a_ng, out=out_rg)
        ok = y_rg is out_rg
        results.append(ok)
        (logger.info if ok else logger.error)(
            "[PASS] " if ok else "[FAIL] " + "no_grad + out requires_grad allowed"
        )

        # 设备类用例: torch_npu 可用时执行（桩按输入设备返回单位阵）
        npu_ready = False
        try:
            import torch_npu  # noqa: F401

            npu_ready = torch.npu.is_available()
        except Exception:  # noqa: BLE001
            npu_ready = False
        if npu_ready:
            a_npu = torch.randn(2, 3, 3, device="npu")
            # 8. device 不匹配: out 在 CPU、计算在 NPU
            _expect_runtime_error(
                results,
                "out device mismatch",
                lambda: npu_inv(a_npu, out=torch.zeros(2, 3, 3)),
                "device",
            )
            # 9. 合法 NPU out: 拷贝并返回
            out_npu = torch.zeros(2, 3, 3, device="npu")
            y_npu = npu_inv(a_npu, out=out_npu)
            ok = y_npu is out_npu and torch.equal(
                out_npu.cpu(), torch.eye(3).unsqueeze(0).expand(2, 3, 3)
            )
            results.append(ok)
            (logger.info if ok else logger.error)(
                "[PASS] " if ok else "[FAIL] " + "valid npu out copy"
            )
        else:
            logger.warning("[SKIP] torch_npu 不可用, 跳过 device 类用例")

    return results


def main():
    impl = _load_impl()
    results = run_cases(impl)
    total = len(results)
    passed = sum(results)
    logger.info("=== %d/%d cases passed ===", passed, total)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
