# Detection precision: native and WASM measurements

Empirical host measurements on 2026-10-17. Native target:
`aarch64-apple-darwin`, stable Rust 1.99.0 (`b940084d7`, LLVM 23.1.1). WASM target:
`wasm32-wasip1` with the repository's `simd128` configuration, executed by Node
v22.22.2 / WASI preview1 on the same host. The CPU model and background
load were not recorded. These results describe these host runs, not a worst-case
execution time or browser/device performance.

Before: `0ff69b9022e15084f1493876c77eaca26fb46567` (`fix-shi-tomasi-numerics`), with only the
benchmark and dependency configuration overlaid. After: the uncommitted
`fix-shi-tomasi-precision` snapshot identified by the source hashes in
[the machine-readable results](precision-2026-10-17.json).

Both revisions use imageproc 0.27.0 and the same lockfile, harness, input bytes,
parameters, compiler, and release settings (opt-level 3, LTO, 1 codegen unit).
Before/after build directories are separate and binary hashes are distinct.
Criterion remains 0.5.1 on both revisions: native defaults are unchanged; WASM
uses `cargo_bench_support` without default features so statistical analysis
runs sequentially without worker threads. Analysis occurs outside timed calls.

Each case uses 50 samples, 1 s warm-up, and 3 s target measurement time. All
builds and tests finished before measurements. Runs are sequential, alternating
before and after per input, first natively and then on WASI. The tables use the
Criterion mean estimate and bootstrap 95% confidence interval consistently; a
console slope estimate can differ from the mean. No measurements were discarded.

Timing covers whole allocating detection calls: gradient generation, products,
filtering, scoring, NMS, sorting, spacing and grid occupancy. Image preparation,
corner reporting and file output are outside timing. Quality is 0.05, minimum
distance 8 px, grid 6x4, and cell budget 5 including existing points. Top-up uses
24 fixed cell-center points. The low-contrast case is the same deterministic
320x240 texture mapped to four grey levels before timing.

## Native results

| Input | API | Before, ms/call [95% CI] | After, ms/call [95% CI] | Mean time change | Corners before / after |
|---|---|---:|---:|---:|---:|
| example_720x405 | plain | 4.804 [4.530, 5.231] | 5.374 [5.341, 5.417] | +11.9% | 54 / 60 |
| example_720x405 | grid | 4.659 [4.536, 4.864] | 5.323 [5.303, 5.351] | +14.2% | 37 / 38 |
| example_720x405 | grid_top_up | 4.608 [4.495, 4.774] | 5.476 [5.377, 5.625] | +18.8% | 33 / 34 |
| texture_320x240 | plain | 1.804 [1.710, 1.935] | 1.690 [1.626, 1.776] | -6.3% | 684 / 678 |
| texture_320x240 | grid | 1.462 [1.452, 1.476] | 1.512 [1.478, 1.559] | +3.4% | 120 / 120 |
| texture_320x240 | grid_top_up | 1.475 [1.453, 1.513] | 1.473 [1.465, 1.484] | -0.1% | 96 / 96 |
| texture_640x480 | plain | 6.977 [6.841, 7.158] | 7.236 [7.099, 7.422] | +3.7% | 2711 / 2727 |
| texture_640x480 | grid | 6.073 [6.011, 6.155] | 6.349 [6.317, 6.383] | +4.5% | 120 / 120 |
| texture_640x480 | grid_top_up | 5.924 [5.897, 5.963] | 6.464 [6.411, 6.523] | +9.1% | 96 / 96 |
| low_contrast_320x240 | plain | 1.186 [1.148, 1.244] | 1.698 [1.685, 1.712] | +43.2% | 0 / 679 |
| low_contrast_320x240 | grid | 1.140 [1.129, 1.152] | 1.527 [1.517, 1.538] | +33.9% | 0 / 120 |
| low_contrast_320x240 | grid_top_up | 1.137 [1.117, 1.170] | 1.522 [1.514, 1.532] | +33.9% | 0 / 96 |
| flat_640x480 | plain | 4.557 [4.519, 4.600] | 5.048 [4.910, 5.233] | +10.8% | 0 / 0 |
| flat_640x480 | grid | 4.760 [4.676, 4.893] | 4.999 [4.880, 5.174] | +5.0% | 0 / 0 |
| flat_640x480 | grid_top_up | 4.569 [4.535, 4.612] | 4.982 [4.797, 5.271] | +9.0% | 0 / 0 |

## WASM results (Node/WASI)

| Input | API | Before, ms/call [95% CI] | After, ms/call [95% CI] | Mean time change | Corners before / after |
|---|---|---:|---:|---:|---:|
| example_720x405 | plain | 6.596 [6.570, 6.624] | 5.734 [5.714, 5.755] | -13.1% | 54 / 60 |
| example_720x405 | grid | 6.636 [6.551, 6.764] | 5.835 [5.776, 5.919] | -12.1% | 37 / 38 |
| example_720x405 | grid_top_up | 6.688 [6.564, 6.887] | 6.086 [5.915, 6.336] | -9.0% | 33 / 34 |
| texture_320x240 | plain | 2.021 [1.969, 2.094] | 1.931 [1.837, 2.085] | -4.4% | 684 / 678 |
| texture_320x240 | grid | 1.902 [1.794, 2.054] | 1.641 [1.589, 1.711] | -13.7% | 120 / 120 |
| texture_320x240 | grid_top_up | 1.904 [1.798, 2.067] | 1.605 [1.587, 1.629] | -15.7% | 96 / 96 |
| texture_640x480 | plain | 7.854 [7.802, 7.914] | 7.656 [7.548, 7.843] | -2.5% | 2711 / 2727 |
| texture_640x480 | grid | 7.248 [7.007, 7.662] | 6.693 [6.660, 6.727] | -7.7% | 120 / 120 |
| texture_640x480 | grid_top_up | 6.984 [6.923, 7.052] | 6.618 [6.600, 6.640] | -5.2% | 96 / 96 |
| low_contrast_320x240 | plain | 1.790 [1.756, 1.835] | 1.827 [1.808, 1.854] | +2.1% | 0 / 679 |
| low_contrast_320x240 | grid | 1.782 [1.763, 1.803] | 1.622 [1.608, 1.638] | -9.0% | 0 / 120 |
| low_contrast_320x240 | grid_top_up | 1.738 [1.732, 1.744] | 1.689 [1.661, 1.721] | -2.8% | 0 / 96 |
| flat_640x480 | plain | 7.037 [6.961, 7.142] | 6.445 [6.406, 6.488] | -8.4% | 0 / 0 |
| flat_640x480 | grid | 6.965 [6.938, 6.995] | 6.571 [6.528, 6.622] | -5.7% | 0 / 0 |
| flat_640x480 | grid_top_up | 6.968 [6.931, 7.014] | 7.602 [6.821, 8.664] | +9.1% | 0 / 0 |

## Interpretation and precision checks

The change preserves weak corners but has a target-dependent runtime cost.
For the example image, native mean call time increases by roughly 12-19% across
the three APIs, while Node/WASI mean time decreases by roughly 9-13%. Some
smaller changes are within measurement uncertainty. The WASI flat-image top-up
case has a broad relative-change confidence interval spanning zero; its positive
point estimate does not establish a slowdown.

Low-contrast plain detection changes from 0 to 679 corners. The grid APIs change
from 0 to their available budgets (120 new points for an empty grid, 96 for
top-up). Timings therefore include changed sorting/spacing/output work as well
as different arithmetic and memory traffic; they do not isolate any one cost.
Corner counts match exactly between native and WASI for each revision and all
15 cases. This count comparison alone does not assert cross-target bit identity
of every output score.

The three product planes grow from i16 to i32: an additional
`3 * 640 * 480 * (4 - 2) = 1,843,200` bytes (1.758 MiB) of tensor payload at
640x480. No additional product planes or filtering scratch buffers are added.
Raw Scharr products and window sums are exact within i32; the determinant is
computed exactly in i64. The final score is normalized by `32^2 * sample_count`,
so its intended units are unchanged. Selected points and tuned thresholds can
still change because the previous quantization has been removed.

All 33 tests pass natively and in Node/WASI, including one-grey-level corners,
quadratic contrast scaling, intensity-offset/polarity invariance, border means,
maximum tensor bounds, nearly rank-one tensors, the independent f64/eigensolver
reference, and the existing zero-allocation tracking test. The one-grey-level
corner regression was observed to fail before the precision change.

## Reproduction

See [the benchmark instructions](../README.md). Use this exact harness and the
same dependency overlay on the parent and precision revisions. Build each in a
separate target directory, then run with a common Criterion report directory
per runtime using `--save-baseline before` and `--baseline before`.

```sh
cargo +stable bench --locked --bench features
cargo +stable bench --locked --target wasm32-wasip1 --bench features --no-run
CRITERION_HOME=/absolute/path/to/reports node benches/wasi-runner.cjs /path/to/features.wasm --bench --noplot
```
