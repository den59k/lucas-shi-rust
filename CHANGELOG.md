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
