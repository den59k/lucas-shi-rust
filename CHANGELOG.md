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
  determinant and `lambda_min = determinant / lambda_max` in f64.

Public API signatures and LK tracking are unchanged. Detected positions and
scores can change as a consequence of these corrections; existing detection
thresholds may need retuning. The score normalization is preserved, but eliminating
integer truncation changes responses and selected points again. The three tensor
planes now occupy 4 rather than 2 bytes per pixel per plane. The allocation-free
detector and f32 image paths are separate work.
