#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import glob
import os
import sys

import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))


def _read_output_payload(path, header_magic, np_dtype):
    with open(path, "rb") as f:
        raw = f.read()
    itemsize = np.dtype(np_dtype).itemsize
    if len(raw) >= 8:
        dtype_int = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
        if dtype_int == header_magic:
            dim_count = int(np.frombuffer(raw[4:8], dtype=np.int32)[0])
            header_end = 8 + 8 * dim_count
            if dim_count < 0 or header_end > len(raw):
                return None
            payload = raw[header_end:]
            if len(payload) % itemsize != 0:
                return None
            return np.frombuffer(payload, dtype=np_dtype)
    if len(raw) % itemsize != 0:
        return None
    return np.frombuffer(raw, dtype=np_dtype)


def _decode_complex32(raw_fp16):
    if raw_fp16 is None or len(raw_fp16) % 2 != 0:
        return None
    n_elem = len(raw_fp16) // 2
    out = np.zeros(n_elem, dtype=np.complex64)
    for i in range(n_elem):
        out[i] = complex(float(raw_fp16[2 * i]), float(raw_fp16[2 * i + 1]))
    return out


def compare_data(dtype):
    if dtype == "c64":
        precision = 0.001
        golden_files = sorted(
            glob.glob(os.path.join(curr_dir, "complex64*golden*.bin"))
        )
        output_files = sorted(
            glob.glob(os.path.join(curr_dir, "complex64*output*.bin"))
        )
    elif dtype == "c32":
        precision = 0.01
        golden_files = sorted(
            glob.glob(os.path.join(curr_dir, "complex32*golden*.bin"))
        )
        output_files = sorted(
            glob.glob(os.path.join(curr_dir, "complex32*output*.bin"))
        )
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if not golden_files or not output_files:
        print(f"No golden or output files found for dtype={dtype}")
        sys.exit(1)

    if len(golden_files) != len(output_files):
        print(
            f"FAILED! file list length mismatch (golden={len(golden_files)}, "
            f"output={len(output_files)})"
        )
        sys.exit(1)

    data_same = True
    for gold, out in zip(golden_files, output_files):
        if dtype == "c64":
            tmp_gold = np.fromfile(gold, np.complex64)
            tmp_out = _read_output_payload(out, 16, np.complex64)
        else:
            tmp_gold = _decode_complex32(np.fromfile(gold, np.float16))
            tmp_out = _decode_complex32(_read_output_payload(out, 33, np.float16))

        if (
            tmp_out is None
            or tmp_gold is None
            or len(tmp_out) == 0
            or len(tmp_out) != len(tmp_gold)
        ):
            out_len = 0 if tmp_out is None else len(tmp_out)
            gold_len = 0 if tmp_gold is None else len(tmp_gold)
            print(
                f"FAILED! {out}: invalid or mismatched size "
                f"(out={out_len}, gold={gold_len})"
            )
            data_same = False
            continue

        diff_res = np.isclose(tmp_out, tmp_gold, precision, precision)
        diff_idx = np.where(~diff_res)[0]

        if len(diff_idx) == 0:
            max_abs = np.max(np.abs(tmp_out - tmp_gold))
            max_rel = np.max(np.abs(tmp_out - tmp_gold) / (np.abs(tmp_gold) + 1e-10))
            print(f"PASSED! {out}")
            print(f"Max absolute error: {max_abs}")
            print(f"Max relative error: {max_rel}")
        else:
            print(f"FAILED! {out}")
            for idx in diff_idx[:5]:
                diff_val = tmp_out[idx] - tmp_gold[idx]
                print(
                    f"  index: {idx}, output: {tmp_out[idx]}, "
                    f"golden: {tmp_gold[idx]}, diff: {diff_val}"
                )
            data_same = False

    sys.exit(0 if data_same else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare output with golden data for asdMul"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=["c64", "c32"],
        help="Data type: c64 (complex64) or c32 (complex32)",
    )
    args = parser.parse_args()
    compare_data(args.dtype)
