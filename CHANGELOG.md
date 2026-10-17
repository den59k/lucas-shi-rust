# Changelog

## 0.4.0 (unreleased)

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
- Preserve fractional Scharr gradients and window means in detection. Accumulate
  raw products and 3x3 sums exactly in i32, then normalize the eigenvalue once by
  `32^2 * sample_count`, including image boundaries.
- Preserve small eigenvalues of nearly rank-one tensors using an exact i64
  determinant and `lambda_min = determinant / lambda_max`.

### Performance

- Compute gradient products, 3x3 window sums and scores in one row-by-row pass
  with a three-row ring buffer instead of three full tensor planes. Both box-sum
  passes now stream along rows instead of walking columns.
- Compute scores in f32 after the exact i64 determinant; the formula has no
  remaining cancellation. Rounding can differ from the previous f64 intermediates,
  so nearly equal scores can swap order before spacing.
- Run non-maximum suppression on a score plane with a separable 3x3 maximum and
  collect only positive maxima. When the global maximum lies inside the image it
  is the strongest local maximum, so the relative quality threshold is applied
  while collecting. For a given score plane, selected points are unchanged.
- Complete plain detection on the five benchmark inputs is 3.4-6.2x faster
  natively (aarch64) and 2.1-6.0x faster under Node/WASI in the recorded run.
  See [timings, confidence intervals and output comparisons](benches/results/scoring-performance-2026-10-17.md).
  These figures compare against the full-plane detector before this work and
  were recorded before merging the `DetectorContext` changes below.
- Measured again on `main` after that merge with the OpenCV comparison harness
  (bundled 720×405 frame, Ryzen 9 9900X, medians of 100 calls): detection
  1.03 → 0.77 ms native and 1.61 → 0.76 ms in wasm with `simd128`. The scalar
  wasm build is slightly slower, 3.15 → 3.39 ms. Every corner the crate finds
  now lies within 1.5 px of an OpenCV Scharr corner, up from 97%, and 99% of
  OpenCV's corners have a crate corner within 1.5 px, up from 86%.
- On x86, an AVX2 copy of the detector pipeline is selected at runtime.
- Tracking: window sampling now hoists the bilinear weights, reads contiguous
  row slices and accumulates in lanes the compiler vectorizes, with a
  runtime-selected AVX2 copy on x86. On the bundled 720×405 frame pair, 200
  points, 21 px window, 4 levels, Ryzen 9 9900X: 21.8 → 3.3 ms native,
  24.4 → 4.2 ms in wasm with `simd128`.
- Scalar wasm builds no longer warn about an unused `simd128` import.

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

Detected positions and scores can change as a consequence of these
corrections; existing detection thresholds may need retuning. Eliminating the
integer gradient and filter truncation changes responses and selected points
again.

### Added

- `DetectorContext`: owns the gradient planes, structure-tensor row buffers,
  score plane, candidate list and spacing grid, so `detect` / `detect_grid`
  perform no heap allocation after the first frame. The free detection functions
  are now thin wrappers over it. Covered by the zero-allocation test.
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
