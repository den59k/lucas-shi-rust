# Detection benchmark

Run `cargo +stable bench --locked --bench features` to measure whole calls to
`good_features_to_track` and `good_features_to_track_grid` (empty grid and top-up
with 24 fixed existing points). No benchmark code is used by the library.

Inputs are the bundled 720x405 example image, deterministic smoothed textures at
320x240 and 640x480, a low-contrast 320x240 texture quantized to four grey levels,
and a flat 640x480 image. All image preparation happens before
timing. Every case uses quality 0.05 and minimum distance 8 pixels; grid cases
use 6 columns, 4 rows, and a budget of 5 points per cell, including existing points.

Timing includes gradients, tensor products/filtering, scores, NMS, sorting,
spacing, grid occupancy, allocations, and deallocations. Corner counts are
printed outside timing because numeric corrections change the output and thus
the amount of sorting/spacing work. In particular, the original detector can
return false corners for a flat image. A timing difference is therefore the
whole-call effect of the fixes, not the isolated cost of f64 arithmetic.

For a revision comparison, copy this exact benchmark and its `[[bench]]` manifest
entry into a temporary checkout of the old revision. Use the same Criterion
version on both revisions (currently 0.8.2). Historical reports produced with
0.5.1 remain labeled with that version; do not mix their baselines with new runs.
Build both revisions in separate Cargo target directories with the same stable toolchain, lockfile,
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

## WASM measurements

The same harness also builds for stable `wasm32-wasip1`. The repository target
configuration enables `simd128`; Node's WASI preview1 runtime provides clocks
and report-file access. Criterion's Rayon and plotting defaults are disabled
only on WASM so bootstrap analysis runs without worker threads. Native defaults
are unchanged. No benchmark support is compiled into the library.

```sh
rustup target add wasm32-wasip1 --toolchain stable
cargo +stable bench --locked --target wasm32-wasip1 --bench features --no-run
CRITERION_HOME=/absolute/path/to/wasi-reports node benches/wasi-runner.cjs /path/to/features.wasm --bench --noplot
```

Cargo prints the generated `.wasm` path. Use `--test --noplot` instead of
`--bench --noplot` for a smoke check, or pass Criterion's baseline arguments for
revision comparisons. Build both revisions with identical dependencies and the
same harness. WASI timing here describes Node/V8 on the host; browser runtimes
and devices can have different performance.

## Stage costs and explicit selection

`cargo +stable bench --locked --bench feature_stages` measures gradients, the
fused row-by-row tensor and score pass (`tensor_scores`), NMS, quality filtering,
sorting, spacing, and the whole plain detector separately. `floored_nms_quality`
is the production selection path: it derives the quality floor, runs floored
NMS and applies the quality filter. `-- --test --noplot` runs assertions
and candidate counts without statistical timing. The harness includes the real
private detector functions inside the benchmark binary; the library API and
implementation are unchanged. Its reconstructed output must equal the public
detector output exactly for every input.

Destructive stages use Criterion `iter_batched` with `PerIteration`: input
cloning and returned-buffer destruction occur outside timing. Allocating stages
using `iter` include output destruction. Stages therefore do not sum exactly to
whole-call time: setup changes cache state, timer overhead differs, and buffer
lifetimes differ. Stage measurements describe costs on their stated inputs,
not a causal decomposition obtained by subtracting times.

`controlled/*` supplies the same deterministic synthetic dense scores and the
same resulting candidate lists to both revisions. Input/output fingerprints
and lengths are recorded. This separates selection cost at equal workload from
selection cost on each detector's actual, potentially different responses.
The synthetic fixture is a workload control, not a detector-quality dataset.

The `FLOOR` records sweep absolute thresholds of 0, 0.01, 0.1, 1, 4, and 16
in `(grey levels / pixel)^2`, combined with the existing relative threshold by
`max(absolute, 0.05 * maximum_after_NMS)`. These are exploratory values, not
recommended defaults. They report eligible score pixels (including borders),
qualified local maxima, and final spaced points. A score measures gradient
structure, not the probability that a feature is useful for tracking.

Two benchmark-only NMS variants reject scores below a supplied threshold before
checking neighbors, retaining the original neighbor scores and production tie
behavior. Every threshold's result is asserted equal to normal NMS followed by
that threshold. The relative threshold is obtained from an untimed reference
NMS; its discovery cost is **not** included in these screening measurements.
Consequently, their speedup is a stage-level opportunity, not an implemented
whole-detector speedup. A production implementation must explicitly define how
it obtains the relative maximum, especially when boundary scores exceed interior
scores. Absolute thresholds alone do not have that dependency.

For a reproducible native/WASI comparison against the pre-precision revision:

```sh
python3 benches/compare-feature-stages.py 0ff69b9 /absolute/path/to/new-run-directory
```

The script creates a temporary baseline source tree, overlays the current
manifest, lockfile and benchmarks, and adapts only two names in the benchmark
access wrapper to the old private function names. Historical library sources
stay unchanged. It builds separate binaries, runs parity assertions, then
alternates before/after per input with no concurrent timed jobs. Both sides use
Criterion 0.8.2, 30 samples, 0.3 s warm-up, 1 s target measurement and 10,000
bootstrap resamples. JSON results retain means and 95% confidence intervals,
counts, source/binary hashes and environment information. All dependency builds
are offline, so dependencies and the WASI target must already be installed.

Since products, window sums and scores were fused into one row-by-row pass and
NMS takes a score plane, the access wrapper no longer maps onto revisions before
that change, so this script cannot build them. Compare such revisions with the
whole-call `features` benchmark as described above.
