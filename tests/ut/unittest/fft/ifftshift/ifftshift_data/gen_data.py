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
import sys

import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))

# Slices processed per chunk; bounds memory even for the 300M-element typical
# shape (RO x CH x PE x SPE = 500 x 24 x 500 x 50).
SLICES_PER_CHUNK = 512


def read_exact(fin, nbytes):
    """Reads exactly nbytes from the binary stdin stream (or raises)."""
    buf = fin.read(nbytes)
    if len(buf) != nbytes:
        raise RuntimeError(f"short stdin read: got {len(buf)} of {nbytes} bytes")
    return buf


def gen_golden(ci, row, col, batch, dist, scale):
    """Golden for the reference CUDA ifftShiftScaleBatch2DKernel:
        dst[s][(r + row // 2) % row][(c + col // 2) % col] = scale * src[s][r][c]
    which equals np.roll(src[s], (shiftR, shiftC), axis=(0, 1)) per slice with
    an element-wise float32 scale multiply. Note: like the repo FftShift (which
    rolls by (n + 1) // 2 per the CUDA fftShiftBatch2DKernel), the shift here
    follows the CUDA reference, not np.fft.ifftshift: for odd dimensions the
    two conventions differ by one, and this kernel is the exact inverse of the
    repo FftShift (ceil + floor = n); for even dimensions all coincide.

    The complex64 input is streamed from stdin and the golden is written chunk
    by chunk, so the peak memory/disk footprint stays around one slice chunk."""
    shift_r = row // 2
    shift_c = col // 2
    scale32 = np.float32(scale)
    slice_elems = row * col
    if dist <= 0:
        dist = slice_elems

    golden_path = os.path.join(curr_dir, f"complex64_golden_{ci}.bin")
    fin = sys.stdin.buffer
    with open(golden_path, "wb") as fout:
        done = 0
        while done < batch:
            n_slices = min(SLICES_PER_CHUNK, batch - done)
            # Streamed input cannot seek, so a chunk consumes whole slice slots:
            # non-final chunks read n_slices * dist elements (the trailing
            # inter-slice pad of the last slot is read and ignored), the final
            # chunk stops exactly at the end of the input.
            if done + n_slices >= batch:
                src_count = (n_slices - 1) * dist + slice_elems
            else:
                src_count = n_slices * dist
            block = np.frombuffer(read_exact(fin, src_count * 8), dtype=np.complex64)
            golden_block = np.empty(n_slices * slice_elems, dtype=np.complex64)
            for s in range(n_slices):
                sl = block[s * dist : s * dist + slice_elems].reshape(row, col)
                sl = np.roll(sl, (shift_r, shift_c), axis=(0, 1))
                if scale != 1.0:
                    scaled = np.empty(sl.shape, dtype=np.complex64)
                    scaled.real = (sl.real * scale32).astype(np.float32)
                    scaled.imag = (sl.imag * scale32).astype(np.float32)
                    sl = scaled
                golden_block[s * slice_elems : (s + 1) * slice_elems] = sl.ravel()
            golden_block.tofile(fout)
            done += n_slices
    print(
        f"golden generated for case {ci}: batch={batch} row={row} col={col} dist={dist} scale={scale}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 6 or len(sys.argv) > 7:
        print(
            "usage: gen_data.py <caseIdx> <row> <col> <batch> <dist> [scale]  (input on stdin)",
            file=sys.stderr,
        )
        sys.exit(2)
    ci = int(sys.argv[1])
    scale = float(sys.argv[6]) if len(sys.argv) == 7 else 1.0
    gen_golden(
        ci,
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        scale,
    )
