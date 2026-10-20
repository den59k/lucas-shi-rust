# Changelog

## 0.6.0 — 2026-10-20

The per-frame tracking step of the 8-bit tracker is 2-3.6x faster, with
the same API plus additive options. No signature changes, but tracked
positions change in their last digits on SIMD targets, because the window
arithmetic is now fixed point there; `TrackKernel::Reference` keeps the
arithmetic of 0.5.

### Changed

- The 8-bit tracker (`TrackerContext`, `calc_optical_flow_ex`,
  `calc_optical_flow_fb`) samples and sums its windows in fixed point on x86
  and x86-64 (SSE2; AVX2 when the CPU has it) and on wasm32 built with
  `simd128`, following the scheme of OpenCV's `calcOpticalFlowPyrLK`:
  14-bit integer bilinear weights, 16-bit patches with 5 fractional bits for
  intensities and 6 for gradients (OpenCV keeps 5; the extra bit removed
  the one avoidable deviation from 0.5 on the bundled pair), and exact
  integer window sums of 16-bit products, from `_mm_madd_epi16` /
  `_mm256_madd_epi16` on x86 and widening multiplies on wasm, which V8
  turns into multiply-accumulates on arm64. Lanes are moved into i64 totals
  before they can overflow, for any 8-bit content; this limits fixed-point
  windows to 127 px. A scalar kernel defines the integers, so fixed-point
  results are bit for bit identical on every target: a golden record of
  the bundled pair is checked natively (scalar, SSE2, AVX2), under WASI
  with and without `simd128`, and the web demo's checksums on a phone in
  Chrome match the native ones. Other targets, and wider windows, keep the
  f32 kernel of 0.5, whose results are unchanged bit for bit.
- NEON kernel: not in 0.6.0. aarch64 keeps the f32 reference kernel under
  `TrackKernel::Auto`; `TrackKernel::FixedPoint` runs the scalar
  fixed-point kernel there.
- Deviation from 0.5 on the bundled 720x405 pair (354 corners, 4 levels,
  30 iterations): on the points both versions track, positions differ by a
  median of 0.00027 px with a 13 px window and 0.00019 px with 21 px, and
  by at most 0.009 / 0.013 px except at two points (4.4 px at 13 px, 1.8 px
  at 21 px) where the 0.5 result itself moves by 1.7-12 px when the input
  point moves by 0.0001 px. Statuses agree on 99.4% / 100% of the points
  with `track` and 98.9% / 99.7% with `track_fb`; the differences are
  iteration-cap and round-trip decisions of borderline points. On
  synthetic shifts, a rotation and an affine warp, positions differ by at
  most 0.01 px and every status agrees. Against OpenCV 4.11 the median
  endpoint difference is 0.0037 px (0.5: 0.0039 px).
- The tracker contexts compute each pyramid level's Scharr gradients once
  per frame, when a pass first needs them, and keep them with the pyramid;
  `advance` moves them along with the last frame's pyramid. With
  `track_fb` this halves the gradient work per frame. Results are
  bit-identical to computing them per pass; the free functions still do.
- `tests/fixtures/regression.txt`: the 8-bit tracking cases name their
  kernel, because `Auto` depends on the target. The f32 reference lines
  are unchanged; new `fixed_track...` lines pin the fixed-point kernel.

### Added

- `TrackOptions` (non-exhaustive, `const` builders, `Default` = previous
  behaviour) with `track_with` and `track_fb_with` on `TrackerContext` and
  `FloatTrackerContext`:
  - `fb_levels`: run the backward pass of the forward-backward check over
    the finest levels only. It tracks the forward results back, seeded at
    the original points, as before. On the bundled pair, compared with the
    full check (fixed point; reference in parentheses): one level accepts
    14 (15) more points and rejects 2 (0) more at 13 px, and accepts 16 (15)
    more at 21 px; two levels accept 5-6 more and reject 0-2 more; three
    levels change at most one point. The occlusion test case is still
    caught with one level.
  - `start_level`: start the forward pass below the top of the pyramid,
    seeded with the prediction, when a motion model is already close.
    The top level or above is the previous behaviour.
  - `kernel`: `TrackKernel::{Auto, FixedPoint, Reference}`, the arithmetic
    of the 8-bit tracker; ignored by `FloatTrackerContext`.

### Performance

The per-frame step of a real-time front-end: `advance` with the new frame,
then `track` (FB off), `track_fb` (FB all) or `track_fb_with` 2 or 1
backward levels, of the strongest corners of synthetic textured frames 3 px
apart, 4 levels, 30 iterations. 0.5.1 and 0.6.0 were run in alternation;
milliseconds per step.

On a phone, Xiaomi 11T (Dimensity 1200, Cortex-A78), Chrome 155, the web
demo's `?bench=1` mode at 360x640, medians over 60 samples of 5 steps after
a two-second spin-up:

| Points, window | FB off 0.5.1 | FB off 0.6.0 | FB all 0.5.1 | FB all 0.6.0 | FB 2 levels | FB 1 level |
|---|---:|---:|---:|---:|---:|---:|
| 44, 13 px | 1.08 | 0.68 | 2.04 | 1.00 | 0.80 | 0.72 |
| 44, 21 px | 1.94 | 0.98 | 3.46 | 1.46 | 1.16 | 1.06 |
| 150, 13 px | 3.26 | 1.54 | 6.24 | 2.66 | 1.94 | 1.76 |
| 150, 21 px | 6.16 | 2.50 | 11.38 | 4.38 | 3.22 | 2.84 |

The cost per point (150 against 44 points) drops 2.5-2.8x; a full
forward-backward step of 150 points with a 21 px window is 2.6x faster, and
4.0x with a single backward level. The result checksums of every case equal
those of a native x86 run.

Ryzen 9 9900X, `benches/tracking.rs` (`regime`), means of the Criterion
estimates of two alternating rounds. Native, AVX2:

| Size | Points, window | FB off 0.5.1 | FB off 0.6.0 | FB all 0.5.1 | FB all 0.6.0 | FB 2 levels | FB 1 level |
|---|---|---:|---:|---:|---:|---:|---:|
| 360x640 | 44, 13 px | 0.267 | 0.139 | 0.493 | 0.193 | 0.159 | 0.149 |
| 360x640 | 44, 21 px | 0.481 | 0.183 | 0.842 | 0.270 | 0.220 | 0.203 |
| 360x640 | 150, 13 px | 0.807 | 0.306 | 1.520 | 0.505 | 0.378 | 0.338 |
| 360x640 | 150, 21 px | 1.539 | 0.466 | 2.817 | 0.805 | 0.589 | 0.534 |
| 640x480 | 44, 13 px | 0.321 | 0.177 | 0.598 | 0.238 | 0.200 | 0.190 |
| 640x480 | 44, 21 px | 0.548 | 0.226 | 1.029 | 0.331 | 0.263 | 0.246 |
| 640x480 | 150, 13 px | 0.910 | 0.345 | 1.707 | 0.574 | 0.433 | 0.392 |
| 640x480 | 150, 21 px | 1.674 | 0.516 | 3.147 | 0.881 | 0.643 | 0.586 |

Node 24 WASI, `simd128`:

| Size | Points, window | FB off 0.5.1 | FB off 0.6.0 | FB all 0.5.1 | FB all 0.6.0 | FB 2 levels | FB 1 level |
|---|---|---:|---:|---:|---:|---:|---:|
| 360x640 | 44, 13 px | 0.337 | 0.203 | 0.650 | 0.318 | 0.246 | 0.222 |
| 360x640 | 44, 21 px | 0.623 | 0.313 | 1.136 | 0.491 | 0.374 | 0.352 |
| 360x640 | 150, 13 px | 1.042 | 0.531 | 2.011 | 0.944 | 0.657 | 0.586 |
| 360x640 | 150, 21 px | 2.020 | 0.879 | 3.808 | 1.566 | 1.120 | 1.003 |
| 640x480 | 44, 13 px | 0.389 | 0.254 | 0.740 | 0.375 | 0.278 | 0.260 |
| 640x480 | 44, 21 px | 0.692 | 0.356 | 1.296 | 0.571 | 0.433 | 0.400 |
| 640x480 | 150, 13 px | 1.169 | 0.603 | 2.189 | 1.091 | 0.743 | 0.659 |
| 640x480 | 150, 21 px | 2.177 | 0.966 | 4.097 | 1.755 | 1.246 | 1.107 |

The earlier `frame_step_640x480` case (lattice points, window 21,
`track_fb` of 150 points) goes from 3.89 to 1.06 ms natively and from 5.10
to 2.19 ms under WASI.

The wasm kernel uses widening multiplies because they are fastest on the
phone. A variant with `i32x4.dot_i16x8_s` over interleaved sample pairs,
which computes the same integers, was 10-16% slower per step on the phone
but 37% faster under V8 on x86-64 (150 points, 21 px, FB all: 0.98 against
1.57 ms under WASI).

Against OpenCV on the bundled pair (`benches/opencv-compare/`, 200 points,
21 px, both pyramids plus tracking, no FB check), tracking takes 1.82 ms in
wasm with `simd128` (0.5.0: 4.21), 0.90 ms natively (3.41), and 5.0 ms on
the phone (0.5.1 in the same session: 13.0; OpenCV.js 14.1). Detection is
unchanged.

### Benchmarks

- `benches/tracking.rs` adds `regime`: `advance` plus `track` / `track_fb`
  / `track_fb_with` (1 and 2 backward levels) of the strongest corners on
  portrait 360x640 and 640x480 frames, 4 levels, windows 13 and 21, 44 and
  150 points.
- The web demo has a benchmark mode, `index.html?bench=1`, which times the
  same step in the browser after spinning the CPU up and reports medians
  and result checksums in `<pre id="bench">` and `window.__lkbench`.
- `benches/opencv-compare/` results for 0.6.0, including a phone run; its
  lockfiles drop the dependencies 0.5.1 moved to `[dev-dependencies]`.

## 0.5.1 — 2026-10-20

No breaking changes.

### Changed

- `image` without default features. The library only uses its buffer types
  (`GrayImage`, `ImageBuffer`, `Luma`, `Pixel`), so dependents, wasm builds
  included, no longer compile every image codec, the AV1 encoder and rayon.
  On `wasm32-unknown-unknown` the library's normal dependency tree shrinks
  from 100 to 23 crates. An application that decodes or encodes images
  through `image` and relied on this crate to enable the codecs now enables
  the features it needs in its own manifest, e.g.
  `image = { version = "0.25", default-features = false, features = ["png"] }`.
- `nalgebra` moved to `[dev-dependencies]`: only the reference eigensolver in
  the tests and the SVD baseline in the benches used it. The library has no
  linear-algebra dependency; the wasm32 normal dependency tree is 6 crates.

### Added

- `TrackerContext::advance` / `FloatTrackerContext::advance`: the per-frame
  form of `prepare`. `advance(frame, levels)` makes the pyramid of the last
  frame the previous-frame pyramid by swapping buffers and builds only the
  pyramid of `frame`, so the caller keeps no copy of the last frame and one
  pyramid is built per frame instead of two. Pyramids and tracking results
  equal those of `prepare(last, frame, levels)` bit for bit, and the steady
  state allocates nothing. A changed level count rebuilds the previous
  pyramid from the last frame; without a last frame, or for a frame of
  another size, both pyramids are built from `frame`.

### Benchmarks

- `benches/tracking.rs` adds `frame_step_640x480`: `prepare` against
  `advance` for the pyramid step alone and with `track_fb` of 150 points.
  On a Ryzen 9 9900X the 4-level 640x480 pyramid step takes 12.7 instead of
  25.3 µs under Node 24/WASI with `simd128`, and 57-72 instead of 100-142 µs
  natively (ranges over four runs). With tracking, the step is 7.8 ms under
  WASI and 4.2-5.1 ms natively either way: tracking dominates, and the
  difference is within the noise.

## 0.5.0 — 2026-10-18

### Breaking changes

- Detection returns `Corner { x, y, score }` instead of `(x, y, score)`
  tuples: `good_features_to_track`, `good_features_to_track_grid`, and
  `detect` / `detect_grid` of `DetectorContext` and `FloatDetectorContext`.
  `Corner::position` gives the tracker input; `From` converts to and from the
  tuple.
- Grid detection takes a `Grid { cols, rows, max_per_cell }` instead of three
  positional `u32` arguments.
- Tracking takes `TrackParams { window_size, max_iterations,
  min_eigen_threshold }` instead of three positional arguments, two of them
  `usize`, which could be swapped unnoticed: `calc_optical_flow_ex`,
  `calc_optical_flow_fb`, and `track` / `track_fb` of `TrackerContext` and
  `FloatTrackerContext`. `fb_threshold` stays a separate argument.
- Remove `calc_optical_flow`, deprecated since 0.3.0.

Migration:

```rust
// 0.4
let fresh = detector.detect_grid(&img, 6, 4, 3, 0.05, 12, &points);
let results = ctx.track_fb(&points, None, 21, 30, DEFAULT_MIN_EIGEN_THRESHOLD, DEFAULT_FB_THRESHOLD);
let positions: Vec<(f32, f32)> = fresh.iter().map(|&(x, y, _)| (x as f32, y as f32)).collect();

// 0.5
let grid = Grid { cols: 6, rows: 4, max_per_cell: 3 };
let fresh = detector.detect_grid(&img, grid, 0.05, 12, &points);
let params = TrackParams {
    window_size: 21,
    max_iterations: 30,
    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
};
let results = ctx.track_fb(&points, None, params, DEFAULT_FB_THRESHOLD);
let positions: Vec<(f32, f32)> = fresh.iter().map(Corner::position).collect();
```

### Fixed

- `TrackerContext::prepare`, `calc_optical_flow_ex` and `calc_optical_flow_fb`
  panic when the two frames or pyramids differ in size. Previously the next
  frame was read with the previous frame's width, which returned wrong
  positions or panicked inside the tracking loop.

### Added

- f32 pipeline for images in physical units, e.g. thermal frames in °C:
  `FloatImageView` (validated borrowed frame: finite values up to 2^24 in
  magnitude), `FloatDetectorContext` with `ScoreThreshold` (absolute threshold
  in (unit / pixel)^2, optional relative part against the strongest corner of
  the whole image), and `FloatTrackerContext` (f32 pyramids, prediction and
  forward-backward check; residuals in the input unit, explicit eigenvalue
  threshold, scale-free singularity test). The f32 detector sums the structure
  tensor in f64 and returns the 8-bit detector's result bit for bit on
  integer-valued frames. See
  [the 8-bit versus f32 measurements](benches/results/float-vs-u8-2026-10-17.md).
- `DetectorCapacity` / `TrackerCapacity` with `with_capacity` and `reserve` on
  all four contexts: buffers are allocated up front, including the largest
  possible candidate count, so calls within the capacity never allocate.
  Pyramid levels reuse their allocation when the frame size changes.
- Bit-exact regression fixtures for 8-bit detection and tracking and for the
  f32 pipeline, identical on aarch64 and wasm32-wasip1, and a
  `rust-toolchain.toml` selecting the stable channel with clippy and rustfmt.

### Changed

- Declare the minimum supported Rust version, 1.88, which `image` 0.25.10
  already requires.

### Benchmarks

- Add `benches/float.rs`: 8-bit versus f32 detection, pyramids, tracking and a
  whole frame step on equal data and on a thermal frame, natively and on WASI.
- The allocation tests count per thread, count `alloc_zeroed`, and cover
  `track_fb`, reserved contexts and varying content and point counts.

### Performance

- Non-maximum suppression tests each row in groups of eight pixels without
  branches, which vectorizes, and visits only the pixels a group keeps. Almost
  every pixel fails the test, so the previous scan mostly paid for branches.
  Selected corners are unchanged. Whole detection calls on the five benchmark
  inputs are 9-38% faster natively (aarch64) and 28-40% faster under
  Node/WASI with `simd128` than 0.4.1; the suppression stage alone is 38-62%
  and 54-77% faster (Apple M2, means of two alternating rounds).
- Measured again on `main` after merging #6 and #7 with the OpenCV comparison
  harness (bundled 720×405 frame, Ryzen 9 9900X, medians of 100 calls):
  detection 0.77 → 0.73 ms native, but 0.76 → 1.10 ms in wasm with `simd128`
  under Node 24 on x86-64 (scalar wasm 3.39 → 4.37 ms). Tracking is unchanged
  at 4.2 ms wasm and 3.4 ms native, and all outputs are identical. The wasm
  slowdown comes with the grouped suppression and is input dependent: on the
  same machine under WASI the synthetic benchmark inputs are 20-29% faster
  than 0.4.1 and the example frame is 42-48% slower. On an arm64 phone
  (Xiaomi 11T, Chrome 155) the same frame is 5% faster than 0.4.1, and the
  crate detects 3.3× faster than OpenCV.js there (2.9 vs 9.9 ms). See
  [the comparison README](benches/opencv-compare/README.md).

## 0.4.1 — 2026-10-18

### Fixed

- Preserve fractional Scharr gradients and window means in detection. Gradients
  were divided by 32 and truncated before squaring, so slopes below one grey
  level per pixel scored zero. Raw products and 3x3 sums are now accumulated
  exactly in i32, and the eigenvalue is normalized once by
  `32^2 * sample_count`, including image boundaries. Score units are unchanged,
  (grey levels / pixel)^2.
- Preserve small eigenvalues of nearly rank-one tensors using an exact i64
  determinant and `lambda_min = determinant / lambda_max`. Rank-one tensors
  score exactly zero.

Detected positions and scores change compared with 0.4.0, because the integer
gradient and filter truncation is gone: a corner with one grey level of
contrast is now detected, scores scale with contrast squared, and detection is
invariant to intensity offset and polarity. Relative quality thresholds keep
working; absolute score thresholds may need retuning. On the bundled 720×405
frame the crate now finds 354 corners instead of 319, every one within 1.5 px
of an OpenCV Scharr corner (was 97%), and 99% of OpenCV's corners have a crate
corner within 1.5 px (was 86%).

### Performance

- Compute gradient products, 3x3 window sums and scores in one row-by-row pass
  with a three-row ring buffer instead of three full tensor planes. Both box-sum
  passes now stream along rows instead of walking columns.
- Compute scores in f32 after the exact i64 determinant; the formula has no
  remaining cancellation. Rounding can differ from the previous intermediates,
  so nearly equal scores can swap order before spacing.
- Run non-maximum suppression on a score plane with a separable 3x3 maximum and
  collect only positive maxima. When the global maximum lies inside the image it
  is the strongest local maximum, so the relative quality threshold is applied
  while collecting. For a given score plane, selected points are unchanged.
- Against 0.4.0 on the OpenCV comparison harness (bundled 720×405 frame,
  Ryzen 9 9900X, medians of 100 calls), detection goes from 1.03 to 0.77 ms
  native and from 1.61 to 0.76 ms in wasm with `simd128`, so the `simd128`
  build now matches native. The scalar wasm build is slightly slower, 3.15 to
  3.39 ms. Tracking is untouched.
- The author's recorded run reports plain detection on the five benchmark
  inputs 3.4-6.2x faster natively (aarch64) and 2.1-6.0x faster under
  Node/WASI; see [timings, confidence intervals and output comparisons](benches/results/scoring-performance-2026-10-17.md).
  Those figures compare against the full-plane detector before this work, not
  against 0.4.0, whose `DetectorContext` detector had already taken part of
  that gain.

### Security

- Update the development dependency imageproc to 0.27.0, which includes fixes for
  [RUSTSEC-2026-0115](https://rustsec.org/advisories/RUSTSEC-2026-0115.html),
  [RUSTSEC-2026-0116](https://rustsec.org/advisories/RUSTSEC-2026-0116.html), and
  [RUSTSEC-2026-0117](https://rustsec.org/advisories/RUSTSEC-2026-0117.html).

### Benchmarks

- Add a low-contrast detection case and a Node WASI runner for the same Criterion
  harness. WASM uses sequential statistical analysis without worker threads;
  native Criterion features remain unchanged.
- Upgrade Criterion from 0.5.1 to 0.8.2 for native and WASM benchmarks. Existing
  recorded comparisons used 0.5.1 on both revisions and retain that provenance.
- Measure detector stages and intermediate candidate counts separately, including
  an equal-input selection control and benchmark-only explicit score-floor/NMS
  experiments. Adapt the comparison runner to full-plane and fused pipelines,
  validate discovered benchmark IDs, and record whole-call and stage results on
  native and WASI without adding profiling code to the library.
- Update the OpenCV comparison results in `benches/opencv-compare/` and the
  README performance table for the exact-scoring detector.

## 0.4.0 — 2026-10-17

### Fixed

- Correct the Shi–Tomasi minimum eigenvalue formula. The square root now applies
  to the discriminant rather than to the trace minus the discriminant.
- Correct the centered separable 3×3 box filter. Each pass uses original neighbor
  values and averages only the available samples at image boundaries.
- Return no corners for flat images or images smaller than 3 pixels in either
  dimension, and exclude zero corner responses.
- Preserve every accepted point in spatial occupancy cells when enforcing the
  minimum distance. Previously, a later point could hide an earlier neighbor.
- Support a zero minimum distance and avoid integer overflow for large distances.

Detected positions and scores can change as a consequence of these
corrections; existing detection thresholds may need retuning. The detector
retains its integer gradient and filter quantization.

### Added

- `DetectorContext`: owns the gradient, structure-tensor and score planes, the
  candidate list and the spacing grid, so `detect` / `detect_grid` perform no
  heap allocation after the first frame. The free detection functions are now
  thin wrappers over it. Covered by the zero-allocation test.
- `CONVERGENCE_EPSILON` documents the per-level Lucas-Kanade stopping threshold.
- `benches/opencv-compare/`: a parity and speed harness against OpenCV, native
  and OpenCV.js.

### Changed

- Lucas-Kanade stops once a step is below 0.01 px (was 0.001), and also when two
  consecutive steps cancel, settling on their midpoint. This matches OpenCV's
  effective criterion. Points that previously hit the iteration cap and came
  back `Diverged` now mostly converge; endpoints of already-converged points
  move by under 0.06 px.
- The tracking window must keep its full bilinear footprint inside the image.
  Positions within one pixel of the right or bottom border, which were
  previously sampled with zero padding, now report `OutOfBounds`.
- The web demo keeps a `DetectorContext` for its grid top-up.

### Performance

Bundled 720×405 frame pair, 200 points, 21 px window, 4 levels, Ryzen 9 9900X:

- Tracking: window sampling now hoists the bilinear weights, reads contiguous
  row slices and accumulates in lanes the compiler vectorizes, with a
  runtime-selected AVX2 copy on x86. 21.8 → 3.3 ms native, 24.4 → 4.2 ms in
  wasm with `simd128`.
- Detection: row-wise box filter with a constant divisor, non-maximum
  suppression directly on a flat score plane instead of a per-pixel tuple
  vector, AVX2 copy on x86. The minimum eigenvalue is evaluated in f32 as
  `2·det / (trace + sqrt(disc))` with the determinant and discriminant exact in
  i32, which is cancellation-free, so the f64 intermediates are gone. 5.1 →
  1.0 ms native, 5.1 → 1.6 ms in wasm. Corner positions are unchanged; scores
  differ from 0.4.0-pre by at most one part in ten million.
- Scalar wasm builds no longer warn about an unused `simd128` import.
