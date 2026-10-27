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
logger = logging.getLogger("torch_sip_fftshift_test")


def randn_complex(shape, device):
    """Generate complex64 random tensor (real/imag generated separately)."""
    re = torch.randn(*shape, dtype=torch.float32, device=device)
    im = torch.randn(*shape, dtype=torch.float32, device=device)
    return torch.complex(re, im)


def reference_FftShift(x, row, col, batch, dist):
    """NumPy reference: dst[s][(r+(row+1)//2)%row][(c+(col+1)//2)%col] = src[s][r][c]"""
    src = x.cpu().numpy()
    shift_r = (row + 1) // 2
    shift_c = (col + 1) // 2
    out = np.empty((batch, row, col), dtype=np.complex64)
    for s in range(batch):
        sl = src[s * dist : s * dist + row * col].reshape(row, col)
        out[s] = np.roll(sl, (shift_r, shift_c), axis=(0, 1))
    return out


def test_fft_shift():
    """执行全部用例并返回是否全部通过。"""
    device = "npu"
    test_cases = [
        (6, 8, 3, 0),
        (7, 9, 2, 0),
        (1, 16, 4, 0),
        (16, 1, 2, 0),
        (4, 6, 3, 32),
        (33, 67, 5, 0),
        (128, 50, 6, 0),
        (64, 48, 8, 0),
        (
            500,
            50,
            12000,
            0,
        ),  # typical MRI shape: RO x CH x PE x SPE = 500 x 24 x 500 x 50
    ]

    all_passed = True
    for row, col, batch, dist in test_cases:
        dist_val = dist if dist > 0 else row * col
        in_elems = (batch - 1) * dist_val + row * col

        x = randn_complex((in_elems,), device)
        y_npu = torch_sip.fft_shift(x, row, col, batch, dist)
        y_ref = reference_FftShift(x, row, col, batch, dist_val)

        # FftShift is a pure permutation: require binary-exact equality.
        y_np = y_npu.cpu().numpy()
        match = np.array_equal(y_np, y_ref)
        status = "PASS" if match else "FAIL"
        if not match:
            all_passed = False
            diff = np.abs(y_np.astype(np.complex128) - y_ref.astype(np.complex128))
            n_diff = int((diff > 0).sum())
            logger.error(
                "  [%s] row=%d col=%d batch=%d dist=%d max_diff=%.6e n_diff=%d",
                status,
                row,
                col,
                batch,
                dist,
                diff.max(),
                n_diff,
            )
        else:
            logger.info(
                "  [%s] row=%d col=%d batch=%d dist=%d (%d elems)",
                status,
                row,
                col,
                batch,
                dist,
                batch * row * col,
            )

    # Test auto-infer (3D input)
    x3d = randn_complex((4, 33, 67), device)
    y_auto = torch_sip.fft_shift(x3d)
    y_manual = torch_sip.fft_shift(x3d.reshape(-1), 33, 67, 4, 0)
    match = np.array_equal(y_auto.cpu().numpy(), y_manual.cpu().numpy())
    status = "PASS" if match else "FAIL"
    if not match:
        all_passed = False
    logger.info("  [%s] auto-infer 3D [4, 33, 67]", status)

    logger.info("=" * 50)
    logger.info(
        "fft_shift torch API: %s", "ALL PASSED" if all_passed else "SOME FAILED"
    )
    return all_passed


def main():
    logger.info("fft_shift torch API test")
    try:
        success = test_fft_shift()
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
