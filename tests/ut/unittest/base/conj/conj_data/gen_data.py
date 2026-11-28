# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))


def load_and_gen_golden_data(filename):
    with open(filename, "rb") as f:
        # 1. 读取dtype
        dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dtype_map = {
            0: "undefined",
            1: np.float32,
            2: np.float16,
            3: np.int32,
            4: np.int64,
            16: np.complex64,
            6: np.uint8,
        }  # 与C++枚举匹配
        np_type = dtype_map[dtype_int]
        # 2. 读取维度数量
        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # 3. 读取维度值
        dims = np.frombuffer(f.read(8 * dim_count), dtype=np.int64)

        # 4. 读取原始数据
        data = np.frombuffer(f.read(), dtype=dtype_map[dtype_int])
        data = data.reshape(dims)

        golden = np.conj(data.astype(np_type))
        golden.astype(np_type).tofile(f"{np_type}_golden_t_conj_{0}.bin")


if __name__ == "__main__":
    filename = curr_dir + "/complex64_input_0.bin"
    load_and_gen_golden_data(filename)
