# Fused Shi–Tomasi scoring and NMS: performance results

Empirical host measurements on 2026-10-17. This compares the already precise
full-plane detector against the subsequent row-pipeline and NMS optimization.
It does not compare against the older gradient-quantized detector.

## Plain detector summary

Mean complete public API call time in ms/call; speedup is before/after.
All thresholds, input bytes and image dimensions are held fixed.

| Input | Native, before → after (speedup) | WASI, before → after (speedup) |
|---|---:|---:|
| example_720x405 | 5.63 → 0.90 (6.2×) | 7.23 → 1.30 (5.5×) |
| texture_320x240 | 1.68 → 0.49 (3.5×) | 1.79 → 0.83 (2.1×) |
| texture_640x480 | 7.42 → 2.19 (3.4×) | 7.54 → 3.64 (2.1×) |
| low_contrast_320x240 | 1.75 → 0.43 (4.1×) | 1.81 → 0.82 (2.2×) |
| flat_640x480 | 4.96 → 0.94 (5.3×) | 6.74 → 1.11 (6.0×) |

## Revision and measurement identity

- Before: precise detector sources equivalent to `e03b82c59478984e0453b8d694e6dc3e732088cd` (`e03b82c`). This run archived `80ac716d8cec31506d5e3eb2d8fb9dd45087372f`; the `src/`, `.cargo/` and `examples/` git tree entries were verified identical to the reachable baseline `e03b82c`.
- After: the frozen performance sources now match `1e6cf8784647042329186022cc580f8ad3b5f860`. They were uncommitted when captured on `80ac716d8cec31506d5e3eb2d8fb9dd45087372f`; hashes, the production diff and harness diff are retained in [the JSON artifact](scoring-performance-2026-10-17.json). The subsequent package-author metadata change does not change the build configuration or detector sources.
- Both sides: identical current Cargo manifest, lockfile, inputs and benchmark harnesses; Criterion 0.8.2 and imageproc 0.27.0.
- Stable Rust 1.99.0 (`b940084d7`, LLVM 23.1.1), release opt-level 3, LTO and one codegen unit.
- Native `aarch64-apple-darwin`; WASM `wasm32-wasip1` with `simd128`, Node v22.22.2 / WASI preview1.
- The same WASI runner forwards `fd_write` through a JS function on both revisions. Direct imports crashed the verbose baseline smoke check before measurement; the forwarding boundary passed it. Clock imports and detector kernels are unchanged. No timing samples existed before this preflight fix; diagnostic logs are retained.
- Sources were frozen into separate trees and compiled in separate target directories. The historical library `src/` was unchanged. The adapter calls the old full-plane functions using their native tuple buffers; no representation conversion is timed.
- All builds, tests and smoke checks completed before measurements. Measurements ran sequentially, native then WASI, before then after per input, first whole calls then stages. No concurrent timed jobs were launched by the runner.
- CPU model unavailable in the sandbox; background load was not controlled. These results apply to these host runs, not browser/device performance or worst-case execution time. No cases or samples were discarded.

The whole-call suite uses 50 samples, 1 s warm-up, 3 s target measurement and
100,000 bootstrap resamples per case. Stage cases use 30 samples, 0.3 s warm-up,
1 s target measurement and 10,000 bootstrap resamples. Tables consistently use
Criterion mean estimates and bootstrap 95% confidence intervals; console slope
estimates can differ. Tables round ms values to three decimals where confidence
intervals are shown; the JSON retains nanosecond precision. Earlier result files
retain their original measurements.

## Complete public API calls

Quality 0.05, minimum distance 8 px. Grid: 6 columns × 4 rows, 5 points per cell.
Top-up reserves 24 fixed existing cell-center points. Timing includes gradient
computation, scores, NMS, sorting, spacing, grid occupancy and buffer allocation/
deallocation. Input preparation, output logging and correctness checks are excluded.

| Runtime | Input | API | Before, ms/call [95% CI] | After, ms/call [95% CI] | Speedup | New points before / after |
|---|---|---|---:|---:|---:|---:|
| native | example_720x405 | plain | 5.635 [5.562, 5.726] | 0.902 [0.887, 0.924] | 6.25× | 60 / 60 |
| native | example_720x405 | grid | 5.551 [5.495, 5.642] | 0.976 [0.908, 1.073] | 5.69× | 38 / 38 |
| native | example_720x405 | grid_top_up | 5.537 [5.502, 5.590] | 0.916 [0.891, 0.950] | 6.05× | 34 / 34 |
| native | texture_320x240 | plain | 1.685 [1.667, 1.708] | 0.487 [0.481, 0.493] | 3.46× | 678 / 678 |
| native | texture_320x240 | grid | 1.517 [1.505, 1.531] | 0.359 [0.355, 0.363] | 4.23× | 120 / 120 |
| native | texture_320x240 | grid_top_up | 1.518 [1.505, 1.535] | 0.416 [0.373, 0.492] | 3.65× | 96 / 96 |
| native | texture_640x480 | plain | 7.420 [7.311, 7.547] | 2.188 [2.172, 2.205] | 3.39× | 2727 / 2727 |
| native | texture_640x480 | grid | 6.571 [6.501, 6.685] | 1.498 [1.472, 1.542] | 4.39× | 120 / 120 |
| native | texture_640x480 | grid_top_up | 6.765 [6.540, 7.131] | 1.475 [1.458, 1.499] | 4.59× | 96 / 96 |
| native | low_contrast_320x240 | plain | 1.753 [1.693, 1.842] | 0.432 [0.425, 0.440] | 4.06× | 679 / 679 |
| native | low_contrast_320x240 | grid | 1.584 [1.547, 1.645] | 0.337 [0.330, 0.346] | 4.70× | 120 / 120 |
| native | low_contrast_320x240 | grid_top_up | 1.581 [1.561, 1.611] | 0.340 [0.327, 0.361] | 4.65× | 96 / 96 |
| native | flat_640x480 | plain | 4.960 [4.902, 5.032] | 0.935 [0.930, 0.941] | 5.30× | 0 / 0 |
| native | flat_640x480 | grid | 4.881 [4.852, 4.920] | 0.950 [0.938, 0.967] | 5.14× | 0 / 0 |
| native | flat_640x480 | grid_top_up | 5.292 [4.994, 5.707] | 0.932 [0.928, 0.936] | 5.68× | 0 / 0 |
| wasi | example_720x405 | plain | 7.226 [6.842, 7.668] | 1.302 [1.292, 1.314] | 5.55× | 60 / 60 |
| wasi | example_720x405 | grid | 6.758 [6.469, 7.079] | 1.310 [1.291, 1.333] | 5.16× | 38 / 38 |
| wasi | example_720x405 | grid_top_up | 6.190 [6.100, 6.304] | 1.236 [1.226, 1.246] | 5.01× | 34 / 34 |
| wasi | texture_320x240 | plain | 1.789 [1.773, 1.808] | 0.835 [0.832, 0.839] | 2.14× | 678 / 678 |
| wasi | texture_320x240 | grid | 1.572 [1.560, 1.587] | 0.633 [0.630, 0.637] | 2.48× | 120 / 120 |
| wasi | texture_320x240 | grid_top_up | 1.580 [1.562, 1.603] | 0.636 [0.631, 0.642] | 2.48× | 96 / 96 |
| wasi | texture_640x480 | plain | 7.535 [7.491, 7.593] | 3.638 [3.625, 3.653] | 2.07× | 2727 / 2727 |
| wasi | texture_640x480 | grid | 6.670 [6.622, 6.730] | 2.761 [2.735, 2.799] | 2.42× | 120 / 120 |
| wasi | texture_640x480 | grid_top_up | 6.758 [6.678, 6.877] | 2.730 [2.705, 2.766] | 2.48× | 96 / 96 |
| wasi | low_contrast_320x240 | plain | 1.807 [1.797, 1.818] | 0.819 [0.815, 0.825] | 2.21× | 679 / 679 |
| wasi | low_contrast_320x240 | grid | 1.622 [1.606, 1.643] | 0.640 [0.633, 0.650] | 2.53× | 120 / 120 |
| wasi | low_contrast_320x240 | grid_top_up | 1.624 [1.614, 1.636] | 0.665 [0.647, 0.693] | 2.44× | 96 / 96 |
| wasi | flat_640x480 | plain | 6.741 [6.601, 6.895] | 1.115 [1.109, 1.121] | 6.05× | 0 / 0 |
| wasi | flat_640x480 | grid | 6.405 [6.373, 6.456] | 1.109 [1.104, 1.116] | 5.77× | 0 / 0 |
| wasi | flat_640x480 | grid_top_up | 6.450 [6.389, 6.530] | 1.114 [1.107, 1.122] | 5.79× | 0 / 0 |

## Output comparison and precision

Native and WASI produce exactly identical complete plain outputs (coordinates,
order and scores) within each revision on all five inputs. All three API corner
counts also match across targets. This is an observed result on these fixtures,
not a general cross-target bit-identity guarantee.

The optimized implementation keeps raw products and window sums exact in i32
and the determinant exact in i64, but computes the remaining eigenvalue expression
in f32. It preserves the score units `(grey levels / pixel)^2`. Small score
rounding differences versus the previous f64 intermediates can change ties, NMS
or spacing choices. The following explicitly compares the measured plain outputs.

| Input | Before / after points | Same coordinate set | Same coordinate order | Maximum score difference on shared points |
|---|---:|---|---|---:|
| example_720x405 | 60 / 60 | True | True | 2 ulp; relative 1.6e-07 |
| texture_320x240 | 678 / 678 | True | True | 2 ulp; relative 1.94e-07 |
| texture_640x480 | 2727 / 2727 | True | True | 3 ulp; relative 2.26e-07 |
| low_contrast_320x240 | 679 / 679 | True | True | 2 ulp; relative 1.22e-07 |
| flat_640x480 | 0 / 0 | True | True | 0 ulp; relative 0 |

Score error above is measured only on shared returned coordinates; it is not a
bound for every image tensor. Empty outputs show zero by convention. No absolute
score floor, image downsampling, feature-budget reduction or input precision
reduction was introduced for these timing comparisons. Detection still allocates;
zero-allocation detector/context work remains separate.

## Intermediate workload

Values below are native; per-revision target counts match. The baseline NMS
retains zero maxima until quality filtering; the optimized NMS collects only
positive maxima. `NMS positives` uses the common positive-only definition.

| Input | Pixels | NMS positives before / after | Qualified before / after | Plain output before / after |
|---|---:|---:|---:|---:|
| example_720x405 | 291,600 | 14,720 / 14,720 | 71 / 71 | 60 / 60 |
| texture_320x240 | 76,800 | 4,238 / 4,238 | 4,189 / 4,189 | 678 / 678 |
| texture_640x480 | 307,200 | 17,006 / 17,006 | 16,455 / 16,455 | 2,727 / 2,727 |
| low_contrast_320x240 | 76,800 | 3,795 / 3,795 | 3,784 / 3,784 | 679 / 679 |
| flat_640x480 | 307,200 | 0 / 0 | 0 / 0 | 0 / 0 |

## Fused computation and production selection

`tensor_scores` measures the complete products/window sums/score pass in each
revision using its native buffers. `production_nms_quality` includes threshold
discovery in the optimized detector, NMS and the final quality filter. The baseline
has no early floor and uses its original NMS plus quality filter.

| Runtime | Input | Stage | Before, ms/stage [95% CI] | After, ms/stage [95% CI] |
|---|---|---|---:|---:|
| native | example_720x405 | tensor_scores | 2.284 [2.273, 2.294] | 0.436 [0.433, 0.439] |
| native | example_720x405 | production_nms_quality | 3.057 [3.027, 3.098] | 0.379 [0.378, 0.381] |
| native | texture_320x240 | tensor_scores | 0.604 [0.598, 0.611] | 0.118 [0.117, 0.120] |
| native | texture_320x240 | production_nms_quality | 0.760 [0.757, 0.763] | 0.152 [0.143, 0.169] |
| native | texture_640x480 | tensor_scores | 2.797 [2.782, 2.815] | 0.461 [0.457, 0.466] |
| native | texture_640x480 | production_nms_quality | 3.208 [3.185, 3.236] | 0.616 [0.611, 0.622] |
| native | low_contrast_320x240 | tensor_scores | 0.790 [0.670, 0.978] | 0.120 [0.117, 0.124] |
| native | low_contrast_320x240 | production_nms_quality | 0.755 [0.752, 0.758] | 0.151 [0.150, 0.152] |
| native | flat_640x480 | tensor_scores | 3.636 [3.293, 4.138] | 0.556 [0.550, 0.562] |
| native | flat_640x480 | production_nms_quality | 2.782 [2.666, 2.966] | 0.521 [0.501, 0.556] |
| wasi | example_720x405 | tensor_scores | 2.674 [2.628, 2.733] | 0.554 [0.544, 0.567] |
| wasi | example_720x405 | production_nms_quality | 2.855 [2.798, 2.923] | 0.491 [0.487, 0.495] |
| wasi | texture_320x240 | tensor_scores | 0.696 [0.694, 0.699] | 0.146 [0.145, 0.146] |
| wasi | texture_320x240 | production_nms_quality | 0.664 [0.659, 0.671] | 0.326 [0.325, 0.328] |
| wasi | texture_640x480 | tensor_scores | 3.015 [3.006, 3.024] | 0.572 [0.569, 0.577] |
| wasi | texture_640x480 | production_nms_quality | 2.887 [2.872, 2.907] | 1.463 [1.455, 1.474] |
| wasi | low_contrast_320x240 | tensor_scores | 0.748 [0.745, 0.750] | 0.145 [0.145, 0.146] |
| wasi | low_contrast_320x240 | production_nms_quality | 0.666 [0.662, 0.671] | 0.356 [0.351, 0.365] |
| wasi | flat_640x480 | tensor_scores | 2.756 [2.735, 2.782] | 0.574 [0.570, 0.579] |
| wasi | flat_640x480 | production_nms_quality | 3.179 [3.162, 3.197] | 0.441 [0.430, 0.461] |

The old tensor pass materializes three full i32 planes and a dense tuple buffer.
The optimized pass streams tensor rows through a ring and produces one f32 score
plane. NMS now uses a separable maximum and collects qualifying positive points.
These are computation and representation changes, not an implicit removal of
low-contrast input. Stage measurements help locate costs, but do not establish an
exact additive attribution of the complete-call speedup.

NMS/quality/sort stages use `iter_batched(PerIteration)`: setup clones and returned
buffer destruction are excluded. The NMS input uses each revision's actual
representation; the optimized closure returns both input and output so neither
destruction is timed. Allocating `iter` stages include output destruction. Cache
state, allocation lifetimes, code context and timer overhead differ from complete
calls. Do not sum stage times or mix stage timing boundaries with old reports.

## Controlled selection

The synthetic control supplies identical score values, coordinates and order.
The setup converts to each revision's native representation outside timing.
All four runs have identical input and final output fingerprints. This controls
score data and candidate workload while retaining the actual NMS algorithm and
buffer representation. It is not a tracking-quality dataset.

| Runtime | Stage | Before, ms/stage [95% CI] | After, ms/stage [95% CI] |
|---|---|---:|---:|
| native | nms | 0.262 [0.255, 0.270] | 0.144 [0.141, 0.147] |
| native | quality | 0.002 [0.002, 0.002] | 0.002 [0.002, 0.002] |
| native | sort | 0.026 [0.025, 0.026] | 0.075 [0.036, 0.135] |
| native | spacing | 0.046 [0.045, 0.048] | 0.047 [0.045, 0.048] |
| wasi | nms | 0.234 [0.232, 0.235] | 0.159 [0.158, 0.160] |
| wasi | quality | 0.008 [0.008, 0.008] | 0.008 [0.008, 0.008] |
| wasi | sort | 0.027 [0.027, 0.028] | 0.027 [0.027, 0.027] |
| wasi | spacing | 0.039 [0.039, 0.039] | 0.040 [0.039, 0.040] |

The JSON also retains exploratory absolute-floor counts and historical scalar
screening microbenchmarks. Those screening variants receive a precomputed
relative threshold and exclude its discovery; they are not used for the complete
API speedups above. The production selection stage includes discovery.

## Validation and reproduction

- Fresh frozen-source builds: 40 tests passed natively; 39 passed and 1 ignored on WASI. The WASI-ignored case is the existing `should_panic` debug bounds test. The allocation-free tracking test passed on both.
- 296 timing records were collected. Expected benchmark IDs were discovered from smoke checks and verified against the saved estimates.
- Staged reconstruction equals both the included production detector and the public library API for each input, revision and target.
- Floored production NMS plus quality equals unfloored NMS plus quality on every benchmark input. Explicit-floor experimental NMS matches ordinary NMS followed by each supplied threshold.
- Control fingerprints, native/WASI complete plain outputs, public API counts and current-source snapshot identity were verified.

```sh
python3 benches/compare-feature-stages.py e03b82c /absolute/path/to/new-run-directory
```

See [benchmark instructions](../README.md#stage-costs-and-explicit-selection).
Raw Criterion reports, build/test logs, snapshots and complete plain outputs
remain at `/var/folders/0y/fqccfvmj0xlb_wp95gqpr3j00000gn/T/lk-scoring-nyz5h2gp`. The accompanying JSON preserves timing summaries, confidence
intervals, counts, output comparison fingerprints, source/binary hashes and tests.
