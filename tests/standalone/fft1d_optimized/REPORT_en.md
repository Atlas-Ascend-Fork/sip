# Ascend950 FFT Optimization Validation and Existing Performance Results

Current scope: horizontally contiguous FP32/complex64, power-of-two N=32–8192; 1D batch=N, and 2D batch=1 N×N square matrices.
All target sizes of C2C in both directions, R2C forward, and C2R inverse are enabled, without falling back due to single-point performance regression. The 2D transforms reuse the original transpose flow.

## Current Branch Validation

- Ascend950, CANN 9.2.0-beta.2. The Release build passes (`NO_WERROR=ON`).
- For the 9 target sizes in 1D and 2D, 4 operator/direction combinations, original and candidate versions, 144/144 groups of random-input smoke tests pass; the maximum relative L2 is 2.762502632e-7, with FP64 NumPy golden and a threshold of 1e-3.
- The actual kernel route selection is verified: both axes of 2D C2C are enabled; in 2D, batch=N on the real axis and batch=N/2+1 on the complex axis are both enabled.
- Host tests cover 524288 1D route-selection combinations, the 2D applicable scope, and UB usage under different core counts; both layout and coefficient boundary tests pass.
- The extended formal tests were stopped at the user's request, and the partial timings already produced are not used in this report; no claim is made that the complete regression suite is complete. The complete cases for in-place and non-target fallback are retained in the test tools, and the current branch has not completed these new device regressions.

## Previously Obtained 1D Performance Results

The following table references independent 1D measurements completed before 2D integration and the removal of the small-size fallback. They are not new results of the current branch, nor the complete 2D time consumption.
The baseline is upstream `5a3b1e31`; batch=N, with 300 ACL Event samples in each of the forward and reverse test orders, and 600 samples merged per version to take the median.
Plan, allocation, and H2D/D2H are excluded, and the profiler is not enabled. In that round of 156 validation groups, 154 had other processes present at startup.
The restored 128-point C2C and 128/256-point real paths do not cite the timings of the original paths at that time as optimization effects; their performance is left for comparison in later versions.

| N | Operator | SIP original μs | Existing optimized version μs | Time reduction |
|---:|---|---:|---:|---:|
|32|c2c forward|7.474|5.052|+32.40%|
|32|c2c inverse|7.512|5.137|+31.62%|
|32|r2c forward|6.786|3.582|+47.21%|
|32|c2r inverse|5.651|3.659|+35.24%|
|64|c2c forward|8.429|4.402|+47.78%|
|64|c2c inverse|8.535|4.237|+50.36%|
|64|r2c forward|7.128|5.180|+27.32%|
|64|c2r inverse|7.060|4.751|+32.69%|
|256|c2c forward|6.568|5.582|+15.02%|
|256|c2c inverse|5.786|6.112|-5.63%|
|512|c2c forward|11.895|7.906|+33.53%|
|512|c2c inverse|12.014|8.535|+28.95%|
|512|r2c forward|7.544|6.519|+13.58%|
|512|c2r inverse|7.907|6.513|+17.63%|
|1024|c2c forward|35.194|18.719|+46.81%|
|1024|c2c inverse|35.610|18.922|+46.86%|
|1024|r2c forward|18.306|13.751|+24.88%|
|1024|c2r inverse|18.365|13.219|+28.02%|
|2048|c2c forward|131.124|64.930|+50.48%|
|2048|c2c inverse|131.184|64.920|+50.51%|
|2048|r2c forward|60.594|34.293|+43.41%|
|2048|c2r inverse|60.072|33.954|+43.48%|
|4096|c2c forward|544.292|234.790|+56.86%|
|4096|c2c inverse|544.022|234.044|+56.98%|
|4096|r2c forward|237.976|139.539|+41.36%|
|4096|c2r inverse|237.516|137.685|+42.03%|
|8192|c2c forward|2494.386|1305.012|+47.68%|
|8192|c2c inverse|2494.043|1304.628|+47.69%|
|8192|r2c forward|943.779|544.499|+42.31%|
|8192|c2r inverse|941.893|533.636|+43.34%|

On shared devices, small sizes fluctuate: the 256-point C2C inverse regressed by 5.63% in that round but improved by 12.14% in an earlier round; the 32-point C2R also had a single-round regression. The current branch is enabled according to the unified scope and does not fall back based on these points.

The raw input summaries, timings, device snapshots, and frozen libraries are saved locally; for reproduction steps, see [README_en.md](README_en.md).
