# fft-2d-c2c-4096x4096 — FFT C2C 2D [4096,4096] Fused Single-Kernel Operator

> **A standalone operator project** refactored from `examples/101_ascend950_fft_c2c_2d_4096_axis0`,
> keeping only the FFT2D (both axes) branch, with no dependency on files outside the repository (CATLASS/TLA are vendored).

## 1. Operator Semantics

The input is `[4096, 4096]` complex64, and a 2D C2C FFT is performed on axes 0 and 1:

- forward (`direction=-1`) is equivalent to `np.fft.fft2(x)`.
- inverse (`direction=+1`) is equivalent to `np.fft.ifft2(x, norm="forward")` (no normalization).

## 2. Solution

The four stages are completed within a single `__mix__(1,2)` kernel, and two transposes are used to absorb the column FFTs:

```text
S1: Row FFT (ex99 engine)   X  -> h   = fft(X, axis=1)
S2: Transpose (64x64 SIMT)     h  -> ws  = h.T
S3: Row FFT (ex99 engine)    ws -> t   = fft(h.T, axis=1) = fft(h, axis=0).T
S4: Transpose (64x64 SIMT)     t  -> out = fft(h, axis=0) = fft2(X)
```

The AIC runs the 16×16×16 GEMM of the row FFT; the AIV runs deinterleave / twiddle / radix-2 and the 64×64 SIMT transpose. `SyncAll<false>` is used between stages.

## 3. Directory Structure

| File | Responsibility |
|---|---|
| `fft_c2c_2d.cpp` | host entry point (arguments, bin I/O, `--perf` timing) |
| `fft_plan.cpp/.h` | Plan/Exec/Destroy (coefficient construction, fused single-launch FFT2D) |
| `fft_tiling.cpp/.h` / `fft_tiling_def.h` | tiling parameters |
| `fft_twiddle.h` / `fft_types.h` | coefficient construction / constants |
| `fft_kernel.h` | fused kernel declarations |
| `fft_kernel_fft.cpp` | row FFT engine + fused four-stage kernel |
| `fft_kernel_transpose_stage.h` | transpose stage (inline, reused by the fused kernel) |
| `fft_kernel_utils.h` / `fft_simt_ops.h` / `fft_radix2_ops.h` | GEMM block / SIMT operators |
| `include/catlass` / `include/tla` | vendored template libraries |
| `helper.hpp` | ACL helper macros |

## 4. Build and Run

```bash
# Build
bash build.sh

# Accuracy (both directions, gate: MARE < 2e-5)
bash run_test.sh <device_id>

# Performance (10 warmup + 10 measured runs, both directions)
bash run_perf.sh <device_id>

# Manual
./build/fft_2d_c2c_4096x4096 [direction] [device_id] [data_dir] [--perf]
```

## 5. Baseline Results

| Item | Value |
|---|---|
| forward (direction=-1) | ~1.59 ms |
| inverse (direction=+1) | ~1.58 ms |
| Accuracy | MARE < 2e-5 (measured ~3.6e-7) |

## 6. Link Dependencies

The `libascend_hal.so` library must be linked (full path: `$ASCEND_HOME_PATH/x86_64-linux/devlib/libascend_hal.so`).
This HAL library is a unified link dependency of Ascend. If it is missing, some misaligned data copies (such as `DataCopyPad<Compact>`) may fail silently.
