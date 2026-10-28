#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
import sys

import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(case):
    """IfftShift: scale == 1 为纯排列、scale != 1 为与 golden 相同的 float32
    乘法路径，均要求 NPU 输出与 numpy golden 二进制一致。"""
    golden_file = os.path.join(curr_dir, "complex64_golden_" + str(case) + ".bin")
    output_file = os.path.join(curr_dir, "complex64_output_" + str(case) + ".bin")

    if not os.path.exists(golden_file):
        print(f"Golden file not found: {golden_file}")
        sys.exit(1)

    if not os.path.exists(output_file):
        print(f"Output file not found: {output_file}")
        sys.exit(1)

    golden = np.fromfile(golden_file, dtype=np.complex64)
    output = np.fromfile(output_file, dtype=np.complex64)

    expected_size = golden.size
    print(f"Expected size: {expected_size}")
    print(f"Golden size: {golden.size}")
    print(f"Output size: {output.size}")

    if output.size != expected_size:
        print(f"ERROR: Size mismatch! Expected {expected_size}")
        sys.exit(1)

    if np.array_equal(output, golden):
        print("PASSED! Output is binary identical to the numpy golden.")
        print("Max absolute error: 0")
        sys.exit(0)

    bad = np.flatnonzero(output != golden)
    total_errors = bad.size
    print("FAILED! Output is not binary identical to the numpy golden:")
    print(
        f"Total mismatches: {total_errors} / {expected_size} ({100 * total_errors / expected_size:.2f}%)"
    )

    for i in range(min(10, total_errors)):
        idx = int(bad[i])
        print(f"  idx={idx}: output={output[idx]}, golden={golden[idx]}")

    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compare NPU IfftShift output with the numpy golden (binary exact)"
    )
    parser.add_argument(
        "--case",
        type=int,
        required=True,
        help="UT case index (suffix of complex64_golden_/complex64_output_ files)",
    )
    args = parser.parse_args()
    compare_data(args.case)


if __name__ == "__main__":
    main()
