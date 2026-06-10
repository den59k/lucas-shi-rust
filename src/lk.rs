use image::{GrayImage, ImageBuffer, Luma};

use crate::utils::fast_gradients::compute_gradients;

/// Default minimum-eigenvalue threshold used by [`calc_optical_flow`].
///
/// The minimum eigenvalue of the per-window spatial gradient matrix is
/// normalized by the window area before comparison, so this threshold is
/// independent of `window_size`. A window flatter than this is reported as
/// [`TrackStatus::LowTexture`].
pub const DEFAULT_MIN_EIGEN_THRESHOLD: f32 = 1e-3;

/// Why a feature point ended up where it did after tracking.
///
/// See [`TrackResult`] for the coordinate convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackStatus {
    /// The iteration converged inside the image; the position is trustworthy.
    Tracked,
    /// The search window left the image (in the previous or the next frame).
    OutOfBounds,
    /// The iteration hit `max_iterations` without converging, or a step
    /// exploded (non-finite or larger than the window).
    Diverged,
    /// The minimum eigenvalue of the spatial gradient matrix fell below the
    /// configured threshold, i.e. the window is too flat to track reliably.
    LowTexture,
    /// The point was tracked forward, but re-tracking it back to the previous
    /// frame did not return close enough to the original position. Only
    /// produced by [`calc_optical_flow_fb`]. A strong occlusion/outlier signal.
    FbInconsistent,
}

/// Default forward-backward round-trip threshold (pixels) for
/// [`calc_optical_flow_fb`].
pub const DEFAULT_FB_THRESHOLD: f32 = 0.7;

/// Per-point result of [`calc_optical_flow_ex`].
///
/// # Coordinate convention
/// Positions are in level-0 (full resolution) pixel coordinates. The origin is
/// the center of the top-left pixel, x grows to the right and y downwards; this
/// matches [`crate::good_features_to_track`] and bilinear sampling throughout
/// the crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackResult {
    /// Tracked position in the next frame.
    pub pos: (f32, f32),
    /// Why tracking ended the way it did.
    pub status: TrackStatus,
    /// Mean absolute photometric residual over the window at the final
    /// position, in 8-bit intensity units.
    ///
    /// Comparable across points regardless of `window_size`; intended for
    /// downstream outlier gating. It is [`f32::INFINITY`] when no residual
    /// could be measured (the window was out of bounds).
    pub error: f32,
}

/// Compute optical flow using the pyramidal Lucas-Kanade method.
///
/// This is a thin wrapper over [`calc_optical_flow_ex`] that discards the
/// per-point diagnostics.
///
/// # Arguments
/// * `prev_pyramid` - Previous frame (pyramid of grayscale)
/// * `curr_pyramid` - Next frame (pyramid of grayscale)
/// * `prev_points` - Feature points to track (in prev frame)
/// * `window_size` - Size of the search window (odd number)
/// * `max_iterations` - Max iterations for correct points on each layer
///
/// # Returns
/// Vector of points on next frame
#[deprecated(
    since = "0.3.0",
    note = "use `calc_optical_flow_ex`, which also returns per-point status and error"
)]
pub fn calc_optical_flow(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iterations: usize,
) -> Vec<(f32, f32)> {
    calc_optical_flow_ex(
        prev_pyramid,
        curr_pyramid,
        prev_points,
        None,
        window_size,
        max_iterations,
        DEFAULT_MIN_EIGEN_THRESHOLD,
    )
    .into_iter()
    .map(|r| r.pos)
    .collect()
}

/// Compute optical flow using the pyramidal Lucas-Kanade method, returning a
/// per-point [`TrackResult`] with status and photometric error.
///
/// # Arguments
/// * `prev_pyramid` - Previous frame (pyramid of grayscale)
/// * `curr_pyramid` - Next frame (pyramid of grayscale)
/// * `prev_points` - Feature points to track (in prev frame, level-0 coordinates)
/// * `predicted` - Optional predicted positions in the next frame, one per
///   `prev_point` (level-0 coordinates). When `Some`, the iteration is seeded
///   with the predicted displacement at the coarsest pyramid level instead of
///   zero, which markedly improves convergence under large inter-frame motion
///   (e.g. gyroscope-predicted feature motion). `None` reproduces the classic
///   zero-initialized behavior. Per-level propagation is unchanged.
/// * `window_size` - Size of the search window (odd number)
/// * `max_iterations` - Max iterations per pyramid level
/// * `min_eigen_threshold` - Windows whose normalized minimum gradient
///   eigenvalue is below this value are reported as [`TrackStatus::LowTexture`].
///   See [`DEFAULT_MIN_EIGEN_THRESHOLD`].
///
/// # Panics
/// Panics if `predicted` is `Some` but its length differs from `prev_points`.
///
/// # Returns
/// One [`TrackResult`] per input point, in the same order.
pub fn calc_optical_flow_ex(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    predicted: Option<&[(f32, f32)]>,
    window_size: usize,
    max_iterations: usize,
    min_eigen_threshold: f32,
) -> Vec<TrackResult> {
    assert_eq!(prev_pyramid.len(), curr_pyramid.len());
    assert!(window_size % 2 == 1, "Window size must be odd");
    if let Some(predicted) = predicted {
        assert_eq!(
            predicted.len(),
            prev_points.len(),
            "predicted must have one entry per prev_point"
        );
    }

    let n_levels = prev_pyramid.len();
    let radius = window_size / 2;
    let n_pixels = window_size * window_size;
    let epsilon = 1e-3;
    let det_epsilon = 1e-6;
    let offsets = build_window_offsets(radius);

    // Total displacement per point, accumulated coarse-to-fine in level-0 units.
    // Seeding it from a prediction makes the coarsest level start at the
    // predicted position; everything else is identical to the zero-init path.
    let mut displacements: Vec<(f32, f32)> = match predicted {
        Some(predicted) => prev_points
            .iter()
            .zip(predicted.iter())
            .map(|((px, py), (gx, gy))| (gx - px, gy - py))
            .collect(),
        None => vec![(0.0, 0.0); prev_points.len()],
    };
    let mut statuses: Vec<TrackStatus> = vec![TrackStatus::Tracked; prev_points.len()];
    let mut errors: Vec<f32> = vec![f32::INFINITY; prev_points.len()];

    let mut prev_patch = vec![0.0f32; n_pixels];
    let mut ix_patch = vec![0.0f32; n_pixels];
    let mut iy_patch = vec![0.0f32; n_pixels];

    // Process levels from top (coarse) to bottom (fine).
    for level in (0..n_levels).rev() {
        let scale = 2f32.powi(level as i32);
        let is_finest = level == 0;

        let prev_img = &prev_pyramid[level];
        let curr_img = &curr_pyramid[level];
        let (grad_x, grad_y) = compute_gradients(prev_img);

        for (idx, (prev_x, prev_y)) in prev_points.iter().enumerate() {
            // Scale the original point for the current level.
            let x = *prev_x / scale;
            let y = *prev_y / scale;

            // Add the current displacement, scaled for this level.
            let mut dx = displacements[idx].0 / scale;
            let mut dy = displacements[idx].1 / scale;

            // The window must stay inside the previous image to build the patch.
            if !in_bounds(prev_img, x, y, radius) {
                statuses[idx] = TrackStatus::OutOfBounds;
                continue;
            }

            // Spatial gradient matrix and cached previous/gradient patches.
            let mut gxx = 0.0f32;
            let mut gxy = 0.0f32;
            let mut gyy = 0.0f32;

            for (i, (ox, oy)) in offsets.iter().enumerate() {
                let sample_x = x + ox;
                let sample_y = y + oy;
                let ix = interpolate_alt(&grad_x, sample_x, sample_y) / 32.0;
                let iy = interpolate_alt(&grad_y, sample_x, sample_y) / 32.0;

                prev_patch[i] = interpolate(prev_img, sample_x, sample_y);
                ix_patch[i] = ix;
                iy_patch[i] = iy;
                gxx += ix * ix;
                gxy += ix * iy;
                gyy += iy * iy;
            }

            // Reject low-texture windows up front (normalized by window area so
            // the threshold does not depend on `window_size`).
            let min_eig = min_eigenvalue(gxx, gxy, gyy) / n_pixels as f32;
            if min_eig < min_eigen_threshold {
                statuses[idx] = TrackStatus::LowTexture;
                if is_finest {
                    errors[idx] =
                        window_error(curr_img, &prev_patch, &offsets, x + dx, y + dy, radius);
                }
                continue;
            }

            let Some((inv_h00, inv_h01, inv_h11)) = invert_2x2(gxx, gxy, gyy, det_epsilon) else {
                statuses[idx] = TrackStatus::LowTexture;
                continue;
            };

            // Refine the displacement at the current level.
            let mut converged = false;
            let mut out_of_bounds = false;
            let mut diverged = false;
            for _ in 0..max_iterations {
                let curr_x = x + dx;
                let curr_y = y + dy;

                if !in_bounds(curr_img, curr_x, curr_y, radius) {
                    out_of_bounds = true;
                    break;
                }

                let mut bx = 0.0f32;
                let mut by = 0.0f32;

                for (i, (ox, oy)) in offsets.iter().enumerate() {
                    let curr = interpolate(curr_img, curr_x + ox, curr_y + oy);
                    let error = prev_patch[i] - curr;
                    bx += ix_patch[i] * error;
                    by += iy_patch[i] * error;
                }

                let ddx = inv_h00 * bx + inv_h01 * by;
                let ddy = inv_h01 * bx + inv_h11 * by;
                dx += ddx;
                dy += ddy;

                // Guard against runaway steps.
                if !dx.is_finite()
                    || !dy.is_finite()
                    || ddx.abs() > window_size as f32
                    || ddy.abs() > window_size as f32
                {
                    diverged = true;
                    break;
                }

                if ddx.abs() < epsilon && ddy.abs() < epsilon {
                    converged = true;
                    break;
                }
            }

            statuses[idx] = if out_of_bounds {
                TrackStatus::OutOfBounds
            } else if diverged || !converged {
                TrackStatus::Diverged
            } else {
                TrackStatus::Tracked
            };

            // Update the total displacement with the current level scale.
            displacements[idx] = (dx * scale, dy * scale);

            if is_finest {
                errors[idx] = if out_of_bounds {
                    f32::INFINITY
                } else {
                    window_error(curr_img, &prev_patch, &offsets, x + dx, y + dy, radius)
                };
            }
        }
    }

    prev_points
        .iter()
        .enumerate()
        .map(|(idx, (x, y))| {
            let (dx, dy) = displacements[idx];
            TrackResult {
                pos: (x + dx, y + dy),
                status: statuses[idx],
                error: errors[idx],
            }
        })
        .collect()
}

/// Track points prev->next, then re-track the results next->prev, and flag any
/// point whose round-trip lands further than `fb_threshold` pixels from where it
/// started as [`TrackStatus::FbInconsistent`].
///
/// This is the standard forward-backward consistency check and is the cheapest
/// reliable way to reject occlusions and ambiguous matches before they poison a
/// downstream pose solve. Points that already failed the forward pass keep their
/// forward status (the round-trip is only evaluated for forward-tracked points).
///
/// Both passes reuse the supplied pyramids. The free-function form still
/// allocates its result vectors internally; for a fully allocation-free
/// steady-state call, use the equivalent [`TrackerContext`] method.
///
/// # Arguments
/// * `prev_pyramid` / `next_pyramid` - frame pyramids (shared by both passes)
/// * `prev_points` - points to track (level-0 coordinates)
/// * `predicted` - optional initial guess for the forward pass, see
///   [`calc_optical_flow_ex`]
/// * `window_size`, `max_iterations`, `min_eigen_threshold` - as in
///   [`calc_optical_flow_ex`]
/// * `fb_threshold` - maximum allowed round-trip distance in pixels; see
///   [`DEFAULT_FB_THRESHOLD`]
#[allow(clippy::too_many_arguments)]
pub fn calc_optical_flow_fb(
    prev_pyramid: &[GrayImage],
    next_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    predicted: Option<&[(f32, f32)]>,
    window_size: usize,
    max_iterations: usize,
    min_eigen_threshold: f32,
    fb_threshold: f32,
) -> Vec<TrackResult> {
    let mut forward = calc_optical_flow_ex(
        prev_pyramid,
        next_pyramid,
        prev_points,
        predicted,
        window_size,
        max_iterations,
        min_eigen_threshold,
    );

    let forward_pos: Vec<(f32, f32)> = forward.iter().map(|r| r.pos).collect();
    let backward = calc_optical_flow_ex(
        next_pyramid,
        prev_pyramid,
        &forward_pos,
        None,
        window_size,
        max_iterations,
        min_eigen_threshold,
    );

    mark_fb_inconsistent(&mut forward, &backward, prev_points, fb_threshold);
    forward
}

/// Flag forward-tracked points whose backward round-trip exceeds the threshold.
/// Factored out so the [`TrackerContext`] path can reuse it without allocating.
fn mark_fb_inconsistent(
    forward: &mut [TrackResult],
    backward: &[TrackResult],
    prev_points: &[(f32, f32)],
    fb_threshold: f32,
) {
    let threshold_sq = fb_threshold * fb_threshold;
    for (idx, result) in forward.iter_mut().enumerate() {
        // Only forward-tracked points are eligible; others keep their status.
        if result.status != TrackStatus::Tracked {
            continue;
        }

        let back = &backward[idx];
        let dx = back.pos.0 - prev_points[idx].0;
        let dy = back.pos.1 - prev_points[idx].1;
        if back.status != TrackStatus::Tracked || dx * dx + dy * dy > threshold_sq {
            result.status = TrackStatus::FbInconsistent;
        }
    }
}

/// Minimum eigenvalue of the symmetric 2x2 matrix `[[a, b], [b, c]]`.
fn min_eigenvalue(a: f32, b: f32, c: f32) -> f32 {
    let trace = a + c;
    let det = a * c - b * b;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    (trace - disc) / 2.0
}

/// Mean absolute photometric residual between the cached previous patch and the
/// next image sampled at `(cx, cy)`. Returns [`f32::INFINITY`] if the window is
/// out of bounds.
fn window_error(
    img: &GrayImage,
    prev_patch: &[f32],
    offsets: &[(f32, f32)],
    cx: f32,
    cy: f32,
    radius: usize,
) -> f32 {
    if !in_bounds(img, cx, cy, radius) {
        return f32::INFINITY;
    }

    let mut sum = 0.0f32;
    for (i, (ox, oy)) in offsets.iter().enumerate() {
        let curr = interpolate(img, cx + ox, cy + oy);
        sum += (prev_patch[i] - curr).abs();
    }
    sum / offsets.len() as f32
}

fn build_window_offsets(radius: usize) -> Vec<(f32, f32)> {
    let mut offsets = Vec::with_capacity((2 * radius + 1) * (2 * radius + 1));

    for j in -(radius as i32)..=radius as i32 {
        for i in -(radius as i32)..=radius as i32 {
            offsets.push((i as f32, j as f32));
        }
    }

    offsets
}

fn invert_2x2(a00: f32, a01: f32, a11: f32, det_epsilon: f32) -> Option<(f32, f32, f32)> {
    let det = a00 * a11 - a01 * a01;
    if det.abs() <= det_epsilon {
        return None;
    }

    let inv_det = 1.0 / det;
    Some((a11 * inv_det, -a01 * inv_det, a00 * inv_det))
}

/// Checks that the window stays within image bounds
fn in_bounds(img: &GrayImage, x: f32, y: f32, radius: usize) -> bool {
    let (w, h) = (img.width() as f32, img.height() as f32);
    x >= radius as f32 && x < w - radius as f32 && y >= radius as f32 && y < h - radius as f32
}

/// Bilinear interpolation of the pixel value
fn interpolate(img: &GrayImage, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let mut sum = 0.0;
    for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let px = img
            .get_pixel_checked(*sx as u32, *sy as u32)
            .map(|p| p[0] as f32)
            .unwrap_or(0.0);

        let wx = if sx == &x0 { 1.0 - dx } else { dx };
        let wy = if sy == &y0 { 1.0 - dy } else { dy };

        sum += px * wx * wy;
    }

    sum
}

fn interpolate_alt(img: &ImageBuffer<Luma<i16>, Vec<i16>>, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let mut sum = 0.0;
    for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let px = img
            .get_pixel_checked(*sx as u32, *sy as u32)
            .map(|p| p[0] as f32)
            .unwrap_or(0.0);

        let wx = if sx == &x0 { 1.0 - dx } else { dx };
        let wy = if sy == &y0 { 1.0 - dy } else { dy };

        sum += px * wx * wy;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::invert_2x2;

    #[test]
    fn invert_2x2_returns_inverse_components() {
        let (inv00, inv01, inv11) = invert_2x2(4.0, 1.0, 3.0, 1e-6).unwrap();

        assert!((inv00 - 3.0 / 11.0).abs() < 1e-6);
        assert!((inv01 + 1.0 / 11.0).abs() < 1e-6);
        assert!((inv11 - 4.0 / 11.0).abs() < 1e-6);
    }

    #[test]
    fn invert_2x2_rejects_singular_matrix() {
        assert!(invert_2x2(1.0, 2.0, 4.0, 1e-6).is_none());
    }
}
