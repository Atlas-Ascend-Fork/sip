# Ascend950 FFT transpose validation

The implementation uses 64x64 tiles, column-major tile traversal, SIMD reordering
and two UB buffer slots. It supports contiguous complex64 tensors whose final
two axes are transposed, positive dimensions up to 32768 and at most INT32_MAX
complex elements including folded batch. Other axes/platforms retain AsStrided.
No experimental configuration variable is required.

Build after sourcing CANN and building/installing SIP:

```bash
bash tests/standalone/fft_transpose/build.sh <source> <install> <bin> <build>
export LD_LIBRARY_PATH=<install>/lib:$LD_LIBRARY_PATH
<bin>/transpose_bench 65 129 3 2 <device> timings.csv
<bin>/transpose_bench 31 33 3 2 <device> timings.csv tensor
<bin>/transpose_bench 31 33 3 2 <device> timings.csv strides
<bin>/transpose_bench 31 33 3 2 <device> timings.csv axes01
<bin>/transpose_bench 32767 3 1 2 <device> timings.csv strides
```

The test compares uint32 bit patterns, verifies output guards and unchanged input,
and exercises raw pointers, Tensor descriptors, explicit contiguous strides and
fallback axes. Timing from these correctness runs is not a performance claim.

Historical C9 validation on Ascend950: 768 target transpose geometries passed,
including N*N, N*(N/2+1) and the reverse for N=32*n, n=1..256, batch=1.
Same-round full 2D FFT Event measurements against edd645fa showed about 8.7%-10.2%
lower latency for 4096/8192 C2C and 5.6% for 8192 R2C. 4096 R2C was about 0.34%
slower. These are historical experimental measurements with shared-device cases,
not fresh performance measurements of this cleaned submission.

Clean-default validation (2026-10-30, source b51da14 based on 2d566e0):
Ascend950 build passed; 11 interface/boundary checks and 6 bitwise smoke cases
passed. The nine power-of-two square sizes 32..8192, batch=1, C2C forward/inverse,
R2C and C2R passed: 36 cases per library, 72 baseline/candidate executions.
Outputs matched baseline edd645fa bitwise and FP64 FFT relative L2 was at most
1.905e-7 (limit 1e-3). All 72 executions shared the device. Two Event samples per
execution only checked valid timing output; they are not a new performance study.
The core/fft and ops/fft trees of edd645fa and 2d566e0 are identical.
Subsequent submission edits only complete license comments and this documentation.

Review hardening: failure to initialize the optional tiled kernel clears its
partial state and preserves AsStrided execution. Tensor dataSize=0 retains the
internal unspecified-capacity convention; a declared capacity below the planned
byte count is rejected, as fallback cannot make an undersized buffer safe.
Null or partially overlapping buffers are rejected for the tiled geometry.
Use the `capacity` test mode with at least two complex elements to check these
rejections and unchanged output, followed by a valid transpose. The test harness
releases acquired ACL resources on both normal and exceptional exits.

Review-fix validation (2026-10-30, source e6485fa): build, 11 interface cases,
6 smoke cases and the capacity rejection test passed. Private link-time fault
injection verified tiling allocation failure, tiling copy failure with release,
and successful reinitialization after failure. Both fallback cases preserved
output bits. Full 2D C2C forward/inverse, R2C and C2R at N=32 and 4096 passed
(16 baseline/candidate executions), with bitwise-identical outputs and relative
L2 <= 1.828e-7. These shared-device checks do not replace formal performance data.
The device kernel and tiling are unchanged by the review fixes.
