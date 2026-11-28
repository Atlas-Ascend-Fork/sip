# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))


def load_and_gen_golden_data(filename):
    with open(filename, "rb") as f:
        dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dtype_map = {
            0: np.float32,
            1: np.float32,
            2: np.float16,
            3: np.int32,
            4: np.int64,
            16: np.complex64,
            6: np.uint8,
        }
        np_type = dtype_map.get(dtype_int, np.float32)
        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dims = np.frombuffer(f.read(8 * dim_count), dtype=np.int64)
        data = np.frombuffer(f.read(), dtype=np_type)
        data = data.reshape(dims)

        # ccopy: golden is identical to input x (input_0)
        golden = data.astype(np_type)
        golden.astype(np_type).tofile(f"{np_type}_golden_t_ccopy_{0}.bin")


if __name__ == "__main__":
    filename = curr_dir + "/complex64_input_0.bin"
    load_and_gen_golden_data(filename)
