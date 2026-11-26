# fft-2d-r2c-4096x4096 — FFT R2C 2D [4096,4096] Fused Single-Kernel Operator

**A standalone operator project** refactored from `examples/85_ascend950_fft_r2c_2d` (self-contained, vendored CATLASS/TLA).

## 1. Operator Semantics

- Input: `[1, 4096, 4096]` float32 (real)
- Output: `[1, 4096, 2049]` complex64 (half spectrum, interleaved `[re,im]`)
- forward only, equivalent to `torch.fft.rfft2(x, norm="backward")`
- Accuracy: MARE ≈ 4.7e-7

## 2. Solution

5 stages within a single `__mix__(1,2)` kernel, with `SyncAll<false>` between stages:

```text
S1 R2C-1D: x -> y[4096,2112]   two-row packing + 16x16x16 C2C + conjugate split
S2 transpose:   y -> z[2112,4096]   64x64 complex tile
S3 C2C-1D: z -> w[2112,4096]   16x16x16 C2C
S4 transpose:   w -> w2[4096,2112]  64x64 complex tile
S5 trimming:  w2 -> out[4096,2049] DataCopyPad<Compact>
```

## 3. Build and Run

```bash
bash build.sh
bash run_test.sh <device_id>   # Accuracy (MARE < 2e-5)
bash run_perf.sh <device_id>   # Performance
```

## 4. Baseline

| Item | Value |
|---|---|
| Accuracy | MARE ≈ 4.74e-7 |
| Performance | ~0.90 ms |

## 5. Link Dependencies

The `libascend_hal.so` library must be linked (full path: `$ASCEND_HOME_PATH/x86_64-linux/devlib/libascend_hal.so`).
The misaligned trimming `DataCopyPad<Compact>` in S5 depends on this HAL. If it is missing, the accuracy is wrong (MARE ~1.4e-1).
