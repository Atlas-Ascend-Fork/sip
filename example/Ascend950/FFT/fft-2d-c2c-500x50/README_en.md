# fft-2d-c2c-500x50

A standalone single-operator project for 2D C2C FFT with independent compilation. The operator performs a 2D FFT on `axes=(1,2)` for `[12000, 500, 50]` complex numbers (`complex64`, interleaved
`[re, im]`), implemented with **AIC GEMM + AIV SIMT fusion**.

| Item | Value |
|---|---|
| Input shape | `[b1=12000, fftN1=500, fftN2=50]` complex64 |
| Operator | 2D C2C FFT (axes 1,2); neither forward nor inverse is normalized |
| Architecture | Ascend950 (`dav-3510`), `__mix__(1,2)` (1 AIC + 2 AIV per block) |
| Measured time | ≈ 5.86 ms (Task Duration) |
| Measured accuracy | MARE ≈ 4e-5 (threshold 1e-4) |

## Directory Structure

```text
fft-2d-c2c-500x50/
├── CMakeLists.txt            # Standalone cmake (find_package(ASC))
├── build.sh                  # One-click build
├── run.sh                    # One-click build + execution + accuracy comparison
├── run_perf.sh               # Performance test
├── gen_data.py               # Data generation + golden + accuracy comparison
├── include/
│   ├── catlass/              # vendored Catlass template library (header-only)
│   └── tla/                  # vendored TLA library
├── fft_types.h               # Constants / digit reversal / tiling macros
├── fft_tiling_def.h          # Device-side tiling POD
├── fft_tiling.h/.cpp         # Host-side tiling computation
├── fft_coeffs.h/.cpp         # Coefficient construction (twiddles + W50)
├── fft_plan.h/.cpp           # Plan creation/execution/destruction
├── fft_kernel.h/.cpp         # mix kernel + launch wrapper
├── fft_kernel_utils.h        # AIC GEMM (RunGemmToUb) + cross-core flags
├── fft_simt_ops.h            # AIV two-stage radix-20/25 column FFT (SIMT)
├── fft_c2c_2d.cpp            # host entry point (main)
├── helper.hpp                # host helpers such as ACL_CHECK (vendored from examples/common)
└── options.hpp               # Gemm/Gemv option structures (dependency of helper.hpp, vendored as above)
```

## Implementation Overview

- **AIC (Cube)**: The 50-point row FFT uses the dense DFT matrix `W50[100,104]` (8-float aligned padded width) to perform two M-half GEMMs. The `NO_SPLIT_SUBBLOCK` Fixpipe directly writes the `[500,104]` tile to the UB of the target AIV.
- **AIV (SIMT)**: The 500-point column FFT uses two-stage register-fused `radix-20 (4×5) + radix-25 (5×5)`, reusing the original twiddle tables and digit reversal. UB traffic is reduced from 4 reads and 3 writes to 2 reads and 1 write.
- Matrix-level ping-pong (AIC/AIV cross-core flags, mode-4), with no GM workspace and no SyncAll.

## Dependencies

- `include/catlass` and `include/tla`: already vendored with the directory. No additional installation is required.
- CANN development kit: before building, run `source <cann>/ascend-toolkit/set_env.sh` (`build.sh` attempts to automatically locate `${HOME}/cann/ascend-toolkit/set_env.sh`).

## Compilation

```bash
bash build.sh
```

Output: `build/fft_2d_c2c_500x50`.

## Run and Accuracy

```bash
bash run.sh [device_id]        # Default device_id=0
```

`run.sh` performs the following steps in sequence: compilation → input/golden generation (`numpy.fft.fft2`, seed 42) → forward/inverse runs → accuracy comparison with the `MARE < 1e-4` threshold.

## Round-trip Mode

`--roundtrip` implements a multi-channel round-trip workload: the data shape is `[RO=500][CH=24][PE=500][SPE=50]` (that is, `[12000, 500, 50]` complex64). On the `CH` calibration channels, the complete chain of `fftshift → forward → inverse → ifftshift(+1/N)` is repeatedly executed on each `[500, 50]` plane, and the overall time consumed and throughput are collected.

```bash
./build/fft_2d_c2c_500x50 500 -1 <device_id> --roundtrip
```

- The inner loop runs `CH = 24` times. Each time, fftshift, one forward + one inverse 2D FFT, and ifftshift (with `1/N` scaling, where `N = PE×SPE = 500×50 = 25000`) are performed on the full `[12000, 500, 50]` data.
- Since the inverse includes `1/N` normalization, the round-trip output restores the original input (not `input × N`).
- Output statistics: `Time / Total FFTs / FFTs/sec / Max diff / Test`

  - `Total FFTs = RO×CH²×2 = 500×24²×2 = 576000`
  - `Max diff` is the maximum point-wise difference between the round-trip output and the original input. The absolute threshold `< 1e-5` determines PASS (no normalization step).

- In this mode, random complex inputs are generated on the host side with a fixed seed (seed 42), and the accuracy check directly compares against the original input.
- Process exit code: 0 for PASSED and non-zero for FAILED, for direct determination by scripts/CI (issue #181).

Measured results (Ascend950PR, 28 AIC / 56 AIV):

```text
Time: 0.306828 s
Total FFTs: 576000 (= RO×CH²×2 = 500×24²×2)
FFTs/sec: 1877274
Max diff: 0.000001
Test: PASSED
```

`Max diff ≈ 1e-6`, lower than the `1e-5` absolute threshold.

## Performance

```bash
bash run_perf.sh [device_id]   # aclrtEvent: 10 warmup + 10 measured runs
```

To reproduce the msprof `Task Duration` (≈5.86 ms):

```bash
source <cann>/ascend-toolkit/set_env.sh
msprof op build/fft_2d_c2c_500x50 500 -1 <device_id> data
```

## Relationship with the 2D C2C Experimental Paths of the asdsip Library

This project is a standalone compiled kernel sample and does not read any environment variables itself. The experimental computation paths provided within the asdsip acceleration library for the same 500×50 shape are controlled by the following environment variables (**all disabled by default**; after being enabled, the performance and numerical behavior may differ from the default paths, and they are for performance experiments and issue locating only):

| Environment Variable | Function | Value |
|---|---|---|
| `ASDSIP_DENSE_GEMM_PASS` | GEMM short-dimension pass (when the short dimension is ≤64, the short-dimension transform pass is replaced by a dense DFT matrix GEMM) | ≤0 disabled (default); >0 enabled |
| `ASDSIP_FUSED_NPASS` | N-dimension fused pass (effective for long dimension 500 and short dimensions [1,52]) | ≤0 disabled (default); >0 enabled |
| `ASDSIP_HYBRID_PASS` | Hybrid kernel (AIC GEMM→UB tiling + AIV fused N transform, replacing the GEMM pass + transpose + N pass; effective only for 500×50; takes precedence when enabled together with `ASDSIP_DENSE_GEMM_PASS`) | ≤0 disabled (default); >0 enabled |
| `ASDSIP_HYBRID_DEBUG` | Hybrid kernel bypass diagnosis | 0 disabled (default); 1=AIC bypass; 2=AIV bypass; 3=bypass on both sides |

Usage (effective for 2D C2C calls linked with the asdsip library; for details, see
`docs/en/Installation_Operation_Guide/environment_variable.md`):

```bash
export ASDSIP_HYBRID_PASS=1     # Enable the hybrid kernel experimental path
export ASDSIP_HYBRID_DEBUG=0    # Keep bypass diagnosis disabled
# Disable the experimental paths: unset ASDSIP_HYBRID_PASS ASDSIP_FUSED_NPASS ASDSIP_DENSE_GEMM_PASS
```
