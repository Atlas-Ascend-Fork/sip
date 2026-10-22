#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file_lists, output_file_lists):
    np_dtype = np.complex64
    precision = 1 / 1000

    # 清单为空说明比较产物缺失，未比较任何数据不能判定为精度通过
    if len(golden_file_lists) == 0 or len(output_file_lists) == 0:
        print(
            f"FAILED! empty file list (golden={len(golden_file_lists)}, output={len(output_file_lists)})"
        )
        sys.exit(1)
    # 清单长度不等时 zip 会静默截断到较短一侧，未配对文件将被漏比
    if len(golden_file_lists) != len(output_file_lists):
        print(
            f"FAILED! file list length mismatch (golden={len(golden_file_lists)}, "
            f"output={len(output_file_lists)})"
        )
        sys.exit(1)

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        # 元素个数为 0 或两侧不一致均属无效比较产物
        if tmp_out.size == 0 or tmp_gold.size == 0 or tmp_out.size != tmp_gold.size:
            print(
                f"FAILED! invalid element count ({os.path.basename(out)}: {tmp_out.size}, "
                f"{os.path.basename(gold)}: {tmp_gold.size})"
            )
            data_same = False
            continue
        diff_res = np.isclose(tmp_out, tmp_gold, precision, precision, True)
        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    sys.exit(0 if data_same else 1)


def get_file_lists():
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process():
    golden_file_lists, output_file_lists = get_file_lists()
    compare_data(golden_file_lists, output_file_lists)


if __name__ == "__main__":
    process()
