# Detection precision: arithmetic cost and selection workload

Empirical measurements on 2026-10-17, `aarch64-apple-darwin`, stable Rust 1.99.0
(`b940084d7`, LLVM 23.1.1), Node v22.22.2 / WASI preview1, `wasm32-wasip1`
with `simd128`. CPU model unavailable in the sandbox; background load was not
recorded or controlled. Results apply to these host runs, not browser/device
performance or worst-case execution time. Some intervals are broad; no samples
or cases were discarded.

## Finding

The precision fix changes both dense per-pixel computation and data-dependent
selection work. Final feature count alone is an inadequate explanation for a
runtime difference. The new non-flat score passes are slower on both measured
runtimes, at equal pixel counts. That pass includes normalization, score-buffer
creation and arithmetic on different tensor values; it does not isolate the
exact determinant alone. Conversely, eliminating quantization removes large
zero-score plateaus and can substantially reduce quality-filter input.
Sorting and spacing extra weak corners adds work, but is not the only change.

Explicit absolute thresholds are a useful selection policy. The measurements
do not establish which features are reliable: no noise model, tracking lifetime
or tracking error was evaluated. Reducing gradient precision would alter the
tensor and ranking before this policy can be applied. Preserve accurate scores
and choose rejection explicitly; an approximate arithmetic path would need its
own accuracy/performance evaluation.

## Method and controls

Before: `0ff69b9022e15084f1493876c77eaca26fb46567`. After: `9e0e39e6ab3c09295efb5f1d4dd810039a5892dd`,
with benchmark-only working-tree changes. Library `src/` was unchanged during
this assessment. Both sides use the current manifest/lockfile (imageproc 0.27.0
and Criterion 0.8.2), identical input generation, release optimization, and
30 samples, 0.3 s warm-up, 1 s target measurement, 10,000 bootstrap resamples.
All builds and smoke checks completed before measurements. Runs were sequential:
native then WASI, before then after per input; no concurrent benchmark jobs.
The historical library source is an untouched git archive. Only two identifiers
in the benchmark access wrapper were adapted to historical private names.

The benchmark includes production private functions in its own module. Both its
whole detector and its staged reconstruction must exactly match the public
library detector. These checks passed on all five inputs for both revisions and
runtimes. Native/WASI per-input counts and reported maxima match per revision.
The controlled fixture has identical input and output fingerprints on all four
binaries: 76,800 score pixels, 2,127 qualified maxima, 797 spaced points.

Means and bootstrap 95% confidence intervals are shown below. This is a new
Criterion-0.8.2 experiment; do not combine its baselines with the earlier
Criterion-0.5.1 report. Full numerical summaries and source/binary hashes are in
[the JSON artifact](selection-costs-2026-10-17.json).

Cloning inputs to destructive stages and dropping their returned buffers are
excluded using `iter_batched(PerIteration)`. Allocating `iter` stages include
output destruction. Cache state, timer overhead, compiler context and buffer
lifetimes differ from a complete call. **Do not sum these stages or claim that
a difference of stage times causally accounts for an exact fraction of a whole
call.** The unchanged gradient stage also varies between runs.

## Actual candidate workload

Quality is 0.05 times the maximum score after NMS; spacing is 8 px. NMS currently
visits all interior pixels and also retains zero-valued local maxima. The quality
filter then excludes zeros and sub-threshold responses.

| Input | Pixels | Positive scores before / after | NMS output before / after | Qualified before / after | Spaced before / after |
|---|---:|---:|---:|---:|---:|
| example_720x405 | 291,600 | 24,281 / 281,785 | 251,606 / 16,974 | 64 / 71 | 54 / 60 |
| texture_320x240 | 76,800 | 76,423 / 76,796 | 4,259 / 4,238 | 4,148 / 4,189 | 684 / 678 |
| texture_640x480 | 307,200 | 305,834 / 307,196 | 17,019 / 17,006 | 16,085 / 16,455 | 2,711 / 2,727 |
| low_contrast_320x240 | 76,800 | 0 / 66,845 | 75,684 / 4,657 | 0 / 3,784 | 0 / 679 |
| flat_640x480 | 307,200 | 0 / 0 | 304,964 / 304,964 | 0 / 0 | 0 / 0 |

For the example, NMS output drops from 251,606 to 16,974 even though final output
grows from 54 to 60. For low contrast, the old score field is entirely zero:
75,684 zero maxima are processed and then discarded. The new path retains 3,784
qualified candidates and 679 spaced points. NMS input dimensions are unchanged;
branch outcomes and neighbor-comparison counts can differ. Actual neighbor
comparison counts were not instrumented, so output counts are not a substitute
for an exact NMS operation count.

## Whole-call time

| Runtime | Input | Before, ms/call [95% CI] | After, ms/call [95% CI] | Mean change |
|---|---|---:|---:|---:|
| native | example_720x405 | 5.276 [5.212, 5.353] | 6.102 [5.773, 6.552] | +15.6% |
| native | texture_320x240 | 1.641 [1.621, 1.664] | 1.642 [1.576, 1.753] | +0.1% |
| native | texture_640x480 | 6.448 [6.360, 6.591] | 7.564 [7.442, 7.704] | +17.3% |
| native | low_contrast_320x240 | 1.127 [1.120, 1.136] | 1.792 [1.677, 1.989] | +59.0% |
| native | flat_640x480 | 4.726 [4.545, 4.965] | 4.665 [4.646, 4.688] | -1.3% |
| wasi | example_720x405 | 6.450 [6.293, 6.665] | 5.529 [5.486, 5.583] | -14.3% |
| wasi | texture_320x240 | 1.764 [1.748, 1.781] | 1.660 [1.632, 1.705] | -5.9% |
| wasi | texture_640x480 | 7.224 [7.182, 7.273] | 7.305 [7.125, 7.579] | +1.1% |
| wasi | low_contrast_320x240 | 1.613 [1.609, 1.617] | 1.680 [1.673, 1.688] | +4.1% |
| wasi | flat_640x480 | 6.498 [6.454, 6.566] | 6.035 [5.989, 6.096] | -7.1% |

## Selected stage times

All values are ms/stage call with 95% CI. Remaining stages and inputs are in JSON.

| Runtime | Input | Stage | Before | After |
|---|---|---|---:|---:|
| native | example_720x405 | tensor_filter | 2.360 [2.289, 2.441] | 1.651 [1.518, 1.819] |
| native | example_720x405 | scores | 0.772 [0.722, 0.841] | 0.970 [0.954, 0.990] |
| native | example_720x405 | nms | 2.786 [2.566, 3.129] | 3.078 [3.047, 3.112] |
| native | example_720x405 | quality | 0.221 [0.217, 0.224] | 0.017 [0.016, 0.017] |
| native | example_720x405 | sort | 0.001 [0.001, 0.001] | 0.001 [0.001, 0.001] |
| native | example_720x405 | spacing | 0.006 [0.006, 0.006] | 0.006 [0.006, 0.006] |
| native | low_contrast_320x240 | tensor_filter | 0.400 [0.398, 0.402] | 0.422 [0.367, 0.511] |
| native | low_contrast_320x240 | scores | 0.158 [0.155, 0.161] | 0.297 [0.278, 0.322] |
| native | low_contrast_320x240 | nms | 0.461 [0.458, 0.464] | 0.730 [0.724, 0.738] |
| native | low_contrast_320x240 | quality | 0.058 [0.057, 0.058] | 0.005 [0.005, 0.005] |
| native | low_contrast_320x240 | sort | 0.000 [0.000, 0.000] | 0.031 [0.031, 0.031] |
| native | low_contrast_320x240 | spacing | 0.000 [0.000, 0.000] | 0.047 [0.045, 0.049] |
| wasi | example_720x405 | tensor_filter | 2.166 [2.146, 2.187] | 1.535 [1.493, 1.597] |
| wasi | example_720x405 | scores | 0.734 [0.716, 0.755] | 1.000 [0.996, 1.005] |
| wasi | example_720x405 | nms | 2.536 [2.327, 2.879] | 2.650 [2.638, 2.662] |
| wasi | example_720x405 | quality | 0.947 [0.916, 0.991] | 0.067 [0.067, 0.068] |
| wasi | example_720x405 | sort | 0.001 [0.001, 0.001] | 0.001 [0.001, 0.001] |
| wasi | example_720x405 | spacing | 0.006 [0.006, 0.006] | 0.007 [0.007, 0.007] |
| wasi | low_contrast_320x240 | tensor_filter | 0.554 [0.551, 0.557] | 0.399 [0.397, 0.401] |
| wasi | low_contrast_320x240 | scores | 0.184 [0.183, 0.185] | 0.307 [0.306, 0.308] |
| wasi | low_contrast_320x240 | nms | 0.519 [0.516, 0.523] | 0.647 [0.643, 0.653] |
| wasi | low_contrast_320x240 | quality | 0.266 [0.264, 0.268] | 0.017 [0.017, 0.017] |
| wasi | low_contrast_320x240 | sort | 0.000 [0.000, 0.000] | 0.040 [0.040, 0.040] |
| wasi | low_contrast_320x240 | spacing | 0.000 [0.000, 0.000] | 0.060 [0.056, 0.065] |

On the native low-contrast input, sorting plus spacing take about 0.078 ms on
3,784 candidates; whole-call mean changes from 1.127 to 1.792 ms. This is evidence
against attributing the entire difference to output features. Scores and NMS also
cost more, while quality filtering costs less. This comparison is diagnostic,
not an additive attribution. The WASI example improves overall despite a slower
score pass, consistent with compensating savings in other stages.

## Equal-workload selection control

Both revisions receive identical synthetic score fields and exactly identical
NMS/quality/sort/spacing inputs. The selection algorithms are unchanged in the
library source. Small residual differences can still arise from generated code,
code layout, cache state and the measurement environment; this experiment does
not identify which mechanism caused each residual difference.

| Runtime | Stage | Before, microseconds [95% CI] | After, microseconds [95% CI] |
|---|---|---:|---:|
| native | nms | 204.673 [197.805, 216.137] | 196.202 [193.685, 199.312] |
| native | quality | 1.592 [1.582, 1.604] | 1.582 [1.577, 1.588] |
| native | sort | 20.955 [20.903, 21.008] | 21.472 [21.089, 22.137] |
| native | spacing | 36.982 [36.824, 37.168] | 38.359 [37.024, 40.542] |
| wasi | nms | 227.126 [226.085, 228.311] | 235.968 [231.163, 243.029] |
| wasi | quality | 7.907 [7.877, 7.940] | 7.931 [7.884, 7.987] |
| wasi | sort | 27.537 [27.227, 27.896] | 27.306 [27.143, 27.497] |
| wasi | spacing | 38.389 [38.211, 38.603] | 38.319 [38.249, 38.394] |

## Explicit absolute score floor

The following is the precise current detector on low contrast. Thresholds have
units `(grey levels / pixel)^2`, combined as
`max(absolute_floor, 0.05 * maximum_after_NMS)`. Its maximum is 0.12890625, so the
current relative floor is approximately 0.00644531. The numbers below are
exploratory policies, not proposed defaults. Native and WASI counts match.

| Absolute floor | Eligible score pixels, including borders | Qualified maxima | Spaced points |
|---:|---:|---:|---:|
| 0 | 47,559 | 3,784 | 679 |
| 0.01 | 42,272 | 3,729 | 679 |
| 0.1 | 41 | 38 | 26 |
| 1 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 |
| 16 | 0 | 0 | 0 |

The same count at floor 0.01 does not assert identical feature coordinates.
Floors up to 4 leave the other non-flat inputs unchanged because their relative
floors already exceed 4. Floor 16 reduces the 320x240 texture from 678 to 669
spaced points, and 640x480 texture from 2,727 to 2,700. The example is unchanged.
These sensitivity differences are why a universal default requires image-scale
and noise assumptions.

## Early screening experiment: opportunity and limit

The benchmark-only NMS variant skips neighborhood checks for zero or below-floor
current scores while preserving every original neighbor score and tie rule.
For all six thresholds, all five inputs, both revisions and both targets, its
output exactly matches ordinary NMS followed by that same threshold.

Current precise path only; values are ms/stage call with 95% CI:

| Runtime | Input | Normal NMS + quality | Early relative screening | Early max(relative, 1) screening |
|---|---|---:|---:|---:|
| native | example_720x405 | 2.970 [2.960, 2.981] | 0.343 [0.341, 0.346] | 0.368 [0.347, 0.401] |
| native | low_contrast_320x240 | 0.744 [0.733, 0.758] | 0.650 [0.646, 0.654] | 0.086 [0.085, 0.089] |
| native | flat_640x480 | 2.223 [2.181, 2.266] | 0.347 [0.345, 0.350] | 0.343 [0.342, 0.344] |
| wasi | example_720x405 | 2.749 [2.728, 2.779] | 0.522 [0.518, 0.527] | 0.524 [0.519, 0.531] |
| wasi | low_contrast_320x240 | 0.666 [0.661, 0.672] | 0.616 [0.612, 0.621] | 0.189 [0.187, 0.190] |
| wasi | flat_640x480 | 3.206 [3.173, 3.251] | 0.419 [0.416, 0.422] | 0.415 [0.413, 0.416] |

**The supplied relative threshold came from untimed reference NMS.** Its discovery
cost is excluded, so these figures do not establish an end-to-end speedup. A
production implementation must preserve or explicitly change the definition of
its relative maximum: simply taking the maximum over the entire score image can
change behavior near boundaries. Absolute-only early rejection, including a
zero-score check, does not require discovering that maximum. A floor of 1 also
deliberately removes every low-contrast point; the saved work changes output.
At floor 0 with the existing relative threshold, the checked outputs are preserved.

## Recommended next change

Keep the precision correction and expose an explicit absolute score floor with
optional relative quality. Implement early absolute/zero rejection before NMS
neighbor comparisons while retaining the dense neighbor scores. Measure that
complete detector separately, including threshold discovery and identical-output
checks where the policy is unchanged. Optimize the dense score pass independently
if its measured cost remains material. Avoid implicit gradient truncation as a
substitute for a documented selection policy. No production optimization or new
selection API was implemented in this assessment.

## Reproduction

```sh
python3 benches/compare-feature-stages.py 0ff69b9 /absolute/path/to/new-run-directory
```

See [method and timing boundaries](../README.md#stage-costs-and-explicit-selection).
Raw Criterion reports and logs for this run remain in the temporary directory
`/var/folders/0y/fqccfvmj0xlb_wp95gqpr3j00000gn/T/lk-stages-dqlxu38e`. Source and binary hashes are retained in the JSON artifact.
