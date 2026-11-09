# Ascend950 mixed FFT validation

The common core serves contiguous FP32/complex64 1D transforms and the 1D passes
of batch-one square 2D transforms. It is selected by default for non-power-of-two
lengths divisible by 32, up to 8192, and row batches from 1 to 8192. Real transforms
of length at most 1024 retain the existing DFT route. Other platforms, strided
passes, 3D transforms, external backends and out-of-range inputs retain their
existing selection. Exact aliasing uses a temporary output and copy-back;
partially overlapping buffers are rejected. The public transpose is unchanged.

The factor plan uses a binary prefix followed by ascending odd factors, reusing
the same stage for repeated factors. Radix 3, 5 and 7 use SIMD butterflies; larger
odd factors use a shared symmetric stage with single or paired outputs. Paired
outputs are selected for lengths of at least 1824. There is no per-prime top-level
implementation or experimental configuration switch.

After sourcing CANN and building/installing SIP, build the tests with:

```bash
bash tests/standalone/fft_mixed/build.sh <source> <install> <bin>
python3 tests/standalone/fft_mixed/run.py --candidate <install> \
    --binary <bin>/fft_bench --device <device> --output <new-results> --smoke
```

The build runs 768 factor-plan/selection checks and 11 negative checks, and
compiles the existing `fft1d_optimized/bench.cpp` public tensor-API benchmark.
Python validation reuses its FP64 golden generator and device snapshot helpers;
NumPy is required. Use `--alias` for exact in-place validation. Every execution
checks the input, output size/finiteness, FP64 relative L2 (limit 1e-3), complete
ACL Event samples and the candidate's actual core selection.

`cases.json` contains 30 configurations: 15 lengths, 18 C2C forward/inverse and
12 real transforms, with row batches 1, 7, 64 and representative 2D axis counts.
They cover single, repeated and combined odd factors. Upstream 889eafc was
verified to use C2C arch35 or real Stockham for these cases; small real DFT and
unsupported high-prime cases are excluded from the performance comparison.

For performance comparison, additionally pass `--baseline <baseline-install>`
and omit `--smoke`. The runner measures both separately installed libraries in
two reversed-order rounds, with 300 complete Event samples per version and case.
Use a baseline that supports all cases, record its exact revision, and inspect
the saved device snapshots. There is no automatic migration, remote build,
profiler, or queue management. A new output directory is required for each run;
failures and generated evidence are retained outside the source tree.

Validation before repository cleanup used computation revision 2e10b03 against
the original 889eafc route: all 30 configurations were faster in both rounds,
with geometric mean speedup 2.325x and maximum relative L2 4.704e-7. Of the 120
formal requests, 24 started on a shared device, so this is not an exclusive-device
performance guarantee. Separate structured, alias, high-prime and 2D reuse
checks passed. These are historical measurements of that frozen build, not a
new timing claim for the cleaned submission. Detailed experiment archives are
kept outside this repository's proposed changes.
