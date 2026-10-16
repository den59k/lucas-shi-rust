# Detection benchmark

Run `cargo +stable bench --locked --bench features` to measure whole calls to
`good_features_to_track` and `good_features_to_track_grid` (empty grid and top-up
with 24 fixed existing points). No benchmark code is used by the library.

Inputs are the bundled 720x405 example image, deterministic smoothed textures at
320x240 and 640x480, and a flat 640x480 image. All image preparation happens before
timing. Every case uses quality 0.05 and minimum distance 8 pixels; grid cases
use 6 columns, 4 rows, and a budget of 5 points per cell, including existing points.

Timing includes gradients, tensor products/filtering, scores, NMS, sorting,
spacing, grid occupancy, allocations, and deallocations. Corner counts are
printed outside timing because numeric corrections change the output and thus
the amount of sorting/spacing work. In particular, the original detector can
return false corners for a flat image. A timing difference is therefore the
whole-call effect of the fixes, not the isolated cost of f64 arithmetic.

For a revision comparison, copy this exact benchmark and its `[[bench]]` manifest
entry into a temporary checkout of the old revision. Build both revisions in
separate Cargo target directories with the same stable toolchain, lockfile,
release profile, and target settings, so Cargo cannot reuse a stale local-crate
artifact across source trees. Run
the old binary with `--save-baseline before` and the fixed binary with
`--baseline before`, using the same Criterion output directory. The Criterion
baseline name alone does not select a library revision. Do not time builds or
run the binaries concurrently. Record the old commit, the fixed diff, compiler,
machine, confidence intervals, and corner counts alongside the timings.

For example, with two already-built executables:

```sh
CRITERION_HOME=/absolute/path/to/reports /path/to/before --bench --noplot --save-baseline before
CRITERION_HOME=/absolute/path/to/reports /path/to/after --bench --noplot --baseline before
```

Passing the same case filter to both commands lets you alternate revisions per
input rather than completing all old measurements before all new measurements.

Defaults are 50 samples, 1 second of warm-up, and 3 seconds of measurement per
case. These measure warmed throughput on the host; they do not establish a
worst-case execution time or results on another deployment CPU.
