# fft-2d-c2r-4096x4096 — FFT C2R 2D [4096,4096] Operator

**A standalone operator project** refactored from `examples/86_ascend950_fft_c2r_2d` (self-contained, vendored CATLASS/TLA).

## 1. Operator Semantics

- Input: `[batch, 4096, 2049]` complex64 (half spectrum, interleaved `[re,im]`)
- Output: `[batch, 4096, 4096]` float32 (real)
- inverse only, equivalent to `torch.fft.irfft2(X, s=(4096,4096), norm="forward")` (no normalization)
- Accuracy: MARE ≈ 3.47e-6

## 2. Solution

Two-stage pipeline (K2 fused into K1), with radix-64 1D IDFT decomposition:

```text
input [batch, N, N/2+1] complex (half spectrum)
  -> K1: Hermitian expand + Row IDFT + fused transpose -> ws1 (transposed complex)
  -> K3: Row IDFT + Real extraction -> output [batch, N, N] real
```

## 3. Build and Run

```bash
bash build.sh
bash run_test.sh <device_id>   # Accuracy (MARE < 2e-5)
bash run_perf.sh <device_id>   # Performance (--benchmark --perf)
```

## 4. Baseline

| Item | Value |
|---|---|
| Accuracy | MARE ≈ 3.40e-6 |
| Performance | ~0.83 ms |

## 5. Link Dependencies

The `libascend_hal.so` library must be linked (full path: `$ASCEND_HOME_PATH/x86_64-linux/devlib/libascend_hal.so`).
