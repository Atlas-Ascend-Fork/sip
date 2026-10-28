#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License).
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import logging
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("torch_sip_ifftshift_test")


def randn_complex(shape, device):
    """Generate complex64 random tensor (real/imag generated separately)."""
    re = torch.randn(*shape, dtype=torch.float32, device=device)
    im = torch.randn(*shape, dtype=torch.float32, device=device)
    return torch.complex(re, im)


def reference_IfftShift(x, row, col, batch, dist, scale):
    """NumPy reference: dst[s][(r+row//2)%row][(c+col//2)%col] = scale*src[s][r][c]"""
    src = x.cpu().numpy()
    shift_r = row // 2
    shift_c = col // 2
    out = np.empty((batch, row, col), dtype=np.complex64)
    for s in range(batch):
        sl = src[s * dist : s * dist + row * col].reshape(row, col)
        sl = np.roll(sl, (shift_r, shift_c), axis=(0, 1))
        if scale != 1.0:
            # 与 kernel 相同的 float32 逐实部/虚部乘，保证逐位一致
            re = (sl.real * np.float32(scale)).astype(np.float32)
            im = (sl.imag * np.float32(scale)).astype(np.float32)
            sl = np.empty(sl.shape, dtype=np.complex64)
            sl.real = re
            sl.imag = im
        out[s] = sl
    return out


def test_ifft_shift():
    """执行全部用例并返回是否全部通过。"""
    device = "npu"
    # (row, col, batch, dist, scale)
    test_cases = [
        (6, 8, 3, 0, 1.0),
        (7, 9, 2, 0, 1.0),  # 奇数维度：与 fft_shift 排列不同
        (1, 16, 4, 0, 1.0),
        (16, 1, 2, 0, 1.0),
        (4, 6, 3, 32, 1.0),
        (33, 67, 5, 0, 1.0),
        (128, 50, 6, 0, 1.0),
        (64, 48, 8, 0, 1.0),
        (500, 50, 12000, 0, 1.0),  # typical MRI shape: RO x CH x PE x SPE
        (7, 9, 2, 0, 1.0 / 63),  # 奇数维度 + 折叠 1/(row*col) 归一化
        (500, 50, 12000, 0, 1.0 / 25000),  # MRI 形状 + 逆 FFT 归一化折叠
        (33, 67, 5, 0, -2.5),  # 负 scale
        (3, 10001, 2, 0, 0.5),  # 超宽列 fallback + scale
    ]

    all_passed = True
    for row, col, batch, dist, scale in test_cases:
        dist_val = dist if dist > 0 else row * col
        in_elems = (batch - 1) * dist_val + row * col

        x = randn_complex((in_elems,), device)
        y_npu = torch_sip.ifft_shift(x, row, col, batch, dist, scale)
        y_ref = reference_IfftShift(x, row, col, batch, dist_val, scale)

        # scale=1 为纯排列、scale!=1 为相同 float32 乘法路径：均要求逐位一致
        y_np = y_npu.cpu().numpy()
        match = np.array_equal(y_np, y_ref)
        status = "PASS" if match else "FAIL"
        if not match:
            all_passed = False
            diff = np.abs(y_np.astype(np.complex128) - y_ref.astype(np.complex128))
            n_diff = int((diff > 0).sum())
            logger.error(
                "  [%s] row=%d col=%d batch=%d dist=%d scale=%g max_diff=%.6e n_diff=%d",
                status,
                row,
                col,
                batch,
                dist,
                scale,
                diff.max(),
                n_diff,
            )
        else:
            logger.info(
                "  [%s] row=%d col=%d batch=%d dist=%d scale=%g (%d elems)",
                status,
                row,
                col,
                batch,
                dist,
                scale,
                batch * row * col,
            )

    # Test auto-infer (3D input)
    x3d = randn_complex((4, 33, 67), device)
    y_auto = torch_sip.ifft_shift(x3d)
    y_manual = torch_sip.ifft_shift(x3d.reshape(-1), 33, 67, 4, 0, 1.0)
    match = np.array_equal(y_auto.cpu().numpy(), y_manual.cpu().numpy())
    status = "PASS" if match else "FAIL"
    if not match:
        all_passed = False
    logger.info("  [%s] auto-infer 3D [4, 33, 67]", status)

    # Round-trip identity: ifft_shift(fft_shift(x)) == x (odd dims differ by one)
    x_od = randn_complex((2, 7, 9), device)
    y_rt = torch_sip.ifft_shift(torch_sip.fft_shift(x_od))
    match = np.array_equal(y_rt.cpu().numpy(), x_od.cpu().numpy())
    status = "PASS" if match else "FAIL"
    if not match:
        all_passed = False
    logger.info("  [%s] round-trip ifft_shift(fft_shift(x)) == x on [2, 7, 9]", status)

    logger.info("=" * 50)
    logger.info(
        "ifft_shift torch API: %s", "ALL PASSED" if all_passed else "SOME FAILED"
    )
    return all_passed


def main():
    logger.info("ifft_shift torch API test")
    try:
        success = test_ifft_shift()
        if success:
            logger.info("All tests passed!")
            return 0
        logger.error("Some tests failed.")
        return 1
    except (ImportError, RuntimeError) as e:
        logger.error("Setup failed: %s", e)
        return 1
    except Exception:
        logger.exception("An unhandled exception occurred")
        return 1


if __name__ == "__main__":
    # 仅在脚本最顶层处理进程退出 (G.FNM.07)
    sys.exit(main())
