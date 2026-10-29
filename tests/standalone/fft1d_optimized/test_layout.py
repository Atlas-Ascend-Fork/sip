# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent index/ownership checks for the optimized core's full and partial vector accesses."""

import unittest
import numpy as np


def interleave(a, b):
    x = np.stack((a, b), axis=-1).reshape(-1)
    return x[:64], x[64:]


class LayoutTests(unittest.TestCase):
    def test_q1_store(self):
        for n in (512, 1024, 2048, 4096, 8192):
            for q in range(0, n // 4, 64):
                ys = [4 * np.arange(q, q + 64) + r for r in range(4)]
                ys[0], ys[2] = interleave(ys[0], ys[2])
                ys[1], ys[3] = interleave(ys[1], ys[3])
                ys[0], ys[1] = interleave(ys[0], ys[1])
                ys[2], ys[3] = interleave(ys[2], ys[3])
                np.testing.assert_array_equal(
                    np.concatenate(ys), np.arange(4 * q, 4 * q + 256)
                )

    def test_twiddle_planes_and_bounds(self):
        for n in (512, 1024, 2048, 4096, 8192):
            # The real axis m=N/2 uses 2*m float coefficients, as does C2C.
            table = np.arange(2 * n, dtype=np.float32)
            quarter = 256
            while quarter <= n // 4:
                for power in (1, 2, 3):
                    offset = quarter - 1 + (power - 1) * quarter
                    for q in range(0, n // 4, 64):
                        first = 2 * (offset + q % quarter)
                        self.assertLessEqual(first + 128, 2 * n)
                        span = table[first : first + 128]
                        expected = 2 * (offset + (np.arange(q, q + 64) % quarter))
                        np.testing.assert_array_equal(span[::2], table[expected])
                        np.testing.assert_array_equal(span[1::2], table[expected + 1])
                quarter *= 4


if __name__ == "__main__":
    unittest.main()
