# FFT Multi-row Tiles and Segmented Pipeline

The contiguous 1D native backend arranges the data regions in the UB according to the actual length, increasing the number of rows processed by each tile. Power-of-two lengths retain the original register/radix4 butterflies; mixed lengths retain the dedicated radix3/5/7 butterflies and the symmetric algorithm for larger prime factors, adding the row dimension to the vector index. The twiddle factors keep the complete tables, with no new compression or reconstruction added.

| Branch | Length | Scheduling |
|---|---|---|
| Power-of-two C2C | 32–256 | Forward enlarges tiles when the original pipeline thresholds are satisfied; inverse retains the original tile |
| Power-of-two C2C | 512–4096 | When the original pipeline thresholds are satisfied, multi-row tiles with three rotating data regions |
| Power-of-two R2C/C2R | 32–512 | Enlarged tiles, retaining register scheduling |
| Power-of-two R2C/C2R | 1024–8192 | When the original pipeline thresholds are satisfied, multi-row tiles with three rotating data regions |
| mixed C2C | ≤2944 | Dynamic UB, multi-row butterflies, four data regions with double buffering |
| mixed C2C | 2976–3808 | Complete-table three data regions, two-row tiles |
| mixed C2C | 3840–6720 | Complete-table three data regions, single-row tiles |
| mixed C2C | Valid mixed lengths from 6752 to 8192 | Two data regions with dynamic layout; regions are reused after a single tile completes its output |
| mixed R2C/C2R | ≤5312 | Compact real workspace, dynamic UB, four data regions with double buffering |
| mixed R2C/C2R | 5344–6688 | Compact workspace, complete-table three rotating data regions |
| mixed R2C/C2R | 6720–8192 | Compact workspace, four data regions with single row |

mixed applies only to non-power-of-two lengths divisible by 32 that the original common planner already supports, and retains the original DFT route selection for real N≤1024. The batch range is 1–8192. C2C supports both forward and inverse, and real supports R2C forward/C2R inverse. The 1D calls of the common mixed path and the reuse axes of 2D square matrices uniformly use the new scheduling; the original fixed-capacity row kernels and the new/old scheduling selection parameter are removed. The 2D route selection of power-of-two lengths, the transpose implementation, the external backend selection, and out-of-scope fallback remain unchanged.

The four data regions are the a/b workspaces of two tiles; the three data regions hold the current input, the computation temporary region, and the prefetch of the next tile respectively. The output region is reused only after writing is complete, and the tail tile is trimmed according to the remaining rows per core. mixed uses a 212 KiB budget; the C2C row stride is 2N+128 floats, and the real row stride is N+128 floats.

## Validation

On Ascend950DT, 40 test requests were completed in three rounds (26 formal comparison groups and 14 functional groups, including repeated controls), with 31284 complete ACL Events and 132 outputs in total. All FP64 relative L2 values are <1e-3, with a maximum of 2.0759343391107836e-7. The first round covered 26 groups, the capacity extension 10 groups, and two-region single-row computation and real boundary 4 groups. Results of the same shape from different rounds are not cherry-picked for the best values.

The original 524288 host route-selection combinations and 204 multi-row descriptor combinations pass. For mixed, 741 groups of checks across three pass types for 247 valid lengths verify capacity, alignment, maximum row count, and route selection; the original 768 planning grids and 11 negative cases also pass. Host checks do not replace device validation, and no claim is made that all direction/batch combinations are verified on the device.

| Branch | Segment/Key Boundary | Device Coverage |
|---|---|---|
| Power-of-two C2C register multi-row | ≤256 | 64 and 256 formal; 64/b7 inverse functional |
| Power-of-two C2C three-region | 512–4096 | 512 inverse and 4096 forward formal; 512/b65 functional |
| Power-of-two real register multi-row | ≤512 | 256 R2C formal; 512 R2C functional |
| Power-of-two real three-region | 1024–8192 | 1024 C2R, 4096 R2C, and 8192 C2R formal; 1024/b129 and 4096/b193 functional |
| mixed C2C four-region | ≤2944 | 544, 2016, and 2944 formal; 2944/b129 tail-tile functional |
| mixed C2C three-region two-row | 2976–3808 | 2976 and 3808 inverse formal; 3808/b193 functional |
| mixed C2C three-region single-row | 3840–6720 | 3840 and 6720 formal; 3840/b193 and 6720/b65 functional |
| mixed C2C two-region single-row | Valid mixed lengths from 6752 to 8192 | 6752 and 8160 inverse formal; 8160/b7 forward functional; the two-region interior reuses the dedicated single-row formula |
| mixed real four-region multi-row | ≤5312, excluding the original DFT range | 5312 R2C formal; the previous independent evidence for 2016/2944/5312 in both directions is retained |
| mixed real three-region two-row | 5344–6688 | 5344 C2R and 6688 R2C formal; 5344/b65 and 6688/b193 functional |
| mixed real four-region single-row | Valid mixed lengths from 6720 to 8192 | 6720 C2R formally verified in both experimental rounds; the regression is retained |

For formal performance, each version runs two rounds in reversed order, with 10 warmups and 300 profiler-free ACL Events per round; the table shows the average of the medians of the two rounds. All batches are 8192, and time reduction = (parent - candidate)/parent. Functional items only check accuracy, and performance is not derived from 3 samples.

| Round | Path | N | Parent μs | Candidate μs | Reduction |
|---|---|---:|---:|---:|---:|
| r1 | c2c forward | 64 | 21.109 | 16.717 | 20.802% |
| r1 | c2c forward | 256 | 38.455 | 29.252 | 23.933% |
| r1 | c2c inverse | 512 | 101.890 | 48.736 | 52.168% |
| r1 | c2c forward | 4096 | 928.511 | 406.776 | 56.191% |
| r1 | r2c forward | 256 | 28.632 | 23.144 | 19.166% |
| r1 | c2r inverse | 1024 | 101.672 | 59.931 | 41.055% |
| r1 | r2c forward | 4096 | 411.594 | 236.400 | 42.565% |
| r1 | c2r inverse | 8192 | 475.689 | 475.767 | -0.017% |
| r1 | c2c forward | 544 | 431.898 | 259.519 | 39.912% |
| r1 | c2c inverse | 2016 | 652.170 | 568.783 | 12.786% |
| r1 | c2c inverse | 2944 | 1505.115 | 1362.876 | 9.450% |
| r1 | c2c inverse | 2976 | 1696.740 | 1510.232 | 10.992% |
| r1 | c2c inverse | 3808 | 1834.379 | 1600.467 | 12.752% |
| r1 | r2c forward | 5312 | 2846.343 | 2479.435 | 12.891% |
| r1 | c2r inverse | 5344 | 4888.275 | 4209.933 | 13.877% |
| r1 | r2c forward | 6688 | 2647.704 | 2542.381 | 3.978% |
| r1 | c2r inverse | 6720 | 1487.967 | 1508.922 | -1.408% |
| r2 | c2c forward | 3840 | 1189.127 | 1096.466 | 7.792% |
| r2 | c2c forward | 6720 | 1864.021 | 1789.091 | 4.020% |
| r3 | c2c inverse | 6752 | 10426.089 | 10493.098 | -0.643% |
| r3 | c2c inverse | 8160 | 3758.717 | 3735.701 | 0.612% |

The r1 baseline is frozen commit 52c4b41, whose FFT content is identical to the merged parent 0c65292. r2 retains the power-of-two objects/API library of r1 and adds the mixed two-region; r3 only modifies the two-region interior single-row computation. The device .text hashes of the remaining keys 0/1/2/3/19/21/22/25 stay identical, allowing existing validation of unchanged branches to be reused; the rounds in the table are retained, and no claim is made that the entire table comes from a single final library measurement.

The final 6752 C2C regresses by 0.643%, and 6720 C2R regresses by 1.408% with 1.425% in an independent controlled retest; both are retained under the unified tiling segmentation. The 0.017% slight regression of 8192 C2R goes in different directions in the two rounds. The initial two-region version regressed 8160 by 6.551%, while the final version improved 0.612% over the parent in the same round; the initial results are kept in the experiment archive and are not mixed into the final table. Without profiling, the contribution of specific instructions cannot be quantified from cross-round differences.

Part of the power-of-two gains also comes from connecting the optimized path when batch≠N, which cannot be fully explained as multi-row or triple-buffering gains. The raw inputs/outputs and device metadata stay in the test environment; the PR submits only the source code, test entry points, and this summary. Reproduction uses the existing run.py --multirow, with 21 formal shapes and 13 functional shapes in total; each run uses an independent output directory.

## Complete Replacement of the Common Mixed Path (r4)

The fixed-capacity Rows<128/512/2048/8192>, old keys 0–3, the old real boundary implementation, and the mixed multirow selection parameters are removed. The 1D and the 2D square-matrix reuse axes uniformly select routes by capacity, retaining the single-row butterfly formulas required by the new version. The device .text of the five new keys 19/21/22/25/26 is identical to r3, and the old keys are no longer generated.

7 groups of 2D square matrices were additionally tested on Ascend950DT NPU5/CPU5 with external batch=1. The baseline is the r3 library corresponding to 0a5caaf before this PR update, and the candidate is the library after the complete replacement. 4 formal and 3 functional groups, with 4818 Events and 22 outputs in total, all pass, with a maximum FP64 relative L2 of 2.952e-7. Formal items run two rounds in reversed order, with 10 warmups and 300 complete ACL Events per version per round, without a profiler; no other processes are present on the target card before or after each group, while other cards have tasks, and no claim of exclusive whole-machine use is made.

| Square Matrix | Direction | Before Replacement μs | After Replacement μs | Reduction |
|---|---|---:|---:|---:|
| 544×544 | c2c forward | 77.187 | 53.796 | 30.304% |
| 2016×2016 | c2c inverse | 364.919 | 333.939 | 8.490% |
| 1536×1536 | r2c forward | 161.986 | 134.958 | 16.685% |
| 1536×1536 | c2r inverse | 152.510 | 127.173 | 16.613% |

3840 C2C forward, 6752 C2C inverse, and 5344 R2C are only functionally validated with 3 samples per version, and no performance conclusion is drawn. These items complete the 2D coverage of the three-region/two-region and compact real three-region paths. The existing 1D results are retained under their original rounds and are not cherry-picked with this round.

Reproduction uses the original single-file `run.py --mixed-2d`, mutually exclusive with `--multirow`. The new route checks have been verified against the 22 real logs of this round, including N rows on the real axis and N/2+1 rows on the complex axis. The complete host capacity/planning checks, formatting, Ruff, text/conflict/private-key, codespell, and OAT checks pass.
