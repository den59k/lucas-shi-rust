# Changelog

## Unreleased

### Performance

- Non-maximum suppression tests each row in groups of eight pixels without
  branches, which vectorizes, and visits only the pixels a group keeps. Almost
  every pixel fails the test, so the previous scan mostly paid for branches.
  Selected corners are unchanged. Whole detection calls on the five benchmark
  inputs are 9-38% faster natively (aarch64) and 28-40% faster under
  Node/WASI with `simd128` than 0.4.1; the suppression stage alone is 38-62%
  and 54-77% faster (Apple M2, means of two alternating rounds).

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
