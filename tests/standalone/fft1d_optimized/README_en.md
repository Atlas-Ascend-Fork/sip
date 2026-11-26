# Ascend950 1D FFT Optimization Validation

Default optimization scope: horizontally contiguous layout, FP32/complex64, power-of-two lengths N=32–8192, batch=N.
In the contiguous 1D native backend, C2C N≤4096 and R2C/C2R N≤8192 are extended to 1≤batch≤8192,
with multi-row tiles selected based on UB capacity. C2C N≥512 and real N≥1024 use three rotating data regions when the original pipeline thresholds are satisfied;
smaller lengths retain register scheduling, and small-length C2C inverse retains the original tile. All paths retain the complete existing coefficient layout.
C2C supports both forward and inverse; R2C forward and C2R inverse are enabled, and all target power-of-two sizes use the optimized path without falling back due to performance regression.
Square matrices of the same size with 2D batch=1 are also enabled: the original transpose flow is retained, with batch=N on the real axis and batch=N/2+1 on the half-spectrum complex axis.
Batches, platforms and directions outside the preceding scope, non-square matrices, and 3D plans keep the original route selection. The inverse transform is not normalized.

The implementation reuses the existing register FFT and radix-4 templates, and adds paired conversion of real mirrored frequency points, single-kernel fusion for small sizes,
short-period coefficient caching, contiguous coefficient reading in subsequent stages, and contiguous write-back at the first stage. Only the selected configuration is compiled; there is no runtime tuning switch.

## Build and Run

Build the upstream baseline and this branch separately, and install them into different directories. Run the following in an Ascend950 environment with CANN configured:

```bash
bash tests/standalone/fft1d_optimized/build.sh /path/to/sip /path/to/candidate /path/to/bin
python3 tests/standalone/fft1d_optimized/run.py \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --binary /path/to/bin/fft_bench --device 2 --output /path/to/new-results
```

Python NumPy is required. `--device` uses a verified ACL device ID; the tests do not reset the device.
You can add `--smoke` first to check short executions of all target lengths. Each run uses a new result directory, and failure records are retained.

Host tests check route selection boundaries and UB capacity under different core counts; layout tests check the first-stage interleaving order and coefficient access boundaries.
Device validation covers the three public 1D and 2D interfaces, C2C in both directions, random/zero/impulse/constant/Nyquist inputs, in-place output, and non-target fallback.
In-place cases restore the input before each execution, and the restoration is not counted in Events; non-in-place cases check that the input is not overwritten.

For accuracy, the FP64 NumPy golden is generated from the actual FP32/complex64 inputs, with relative L2 ≤ 1e-3, and correct size and finiteness.
The zero golden requires an error norm ≤ 1e-3. Fallback cases additionally compare that the outputs of the two versions are byte-for-byte identical.
For formal timing, each version is warmed up 10 times, and 300 ACL Event samples are collected in each of the forward and reverse test orders, with 600 samples merged to take the median.
Timing excludes Plan, allocation, and H2D/D2H, and the profiler is not enabled. Device usage, library hashes, and the actual kernel route selection of each group are recorded.
The raw result directory contains device and process information and is kept only in the test environment; the PR contains only the summary report.

For completed device validation and performance results, see [REPORT_en.md](REPORT_en.md).

For the scope, comparison criteria, and supplementary test results of the multi-row and triple-buffering increments, see [MULTIROW_REPORT_en.md](MULTIROW_REPORT_en.md).
To reproduce this limited set of supplementary tests, add `--multirow` to the preceding `run.py` command. The baseline uses the parent version that already includes the original power-of-two and common mixed optimizations.
This option selects only 21 formal comparisons and 13 functional boundaries, and does not run the default full 1D/2D test set.

The 2D supplementary tests after the common mixed path is fully replaced use the same `run.py --mixed-2d`, with 4 formal comparisons and 3 functional checks in total.
The baseline should be the common mixed version before the replacement. It is mutually exclusive with `--multirow`, and does not extend to the default full test set.
