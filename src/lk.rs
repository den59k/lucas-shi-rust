use image::GrayImage;

use crate::features::reserve_total;
use crate::pyramid::{build_pyramid_into, reserve_pyramid};
use crate::utils::fast_gradients::compute_gradients_into;

/// Default minimum-eigenvalue threshold used by [`calc_optical_flow`].
///
/// The minimum eigenvalue of the per-window spatial gradient matrix is
/// normalized by the window area before comparison, so this threshold is
/// independent of `window_size`. A window flatter than this is reported as
/// [`TrackStatus::LowTexture`].
pub const DEFAULT_MIN_EIGEN_THRESHOLD: f32 = 1e-3;

/// Per-level convergence threshold, in pixels of the current pyramid level.
///
/// The iteration at a level stops once both components of the displacement
/// update are below this value, or once two consecutive updates cancel each
/// other (the estimate oscillates around the optimum, which is then taken as
/// their midpoint). This matches the effective stopping precision of OpenCV's
/// `calcOpticalFlowPyrLK`. A point that reaches `max_iterations` without
/// meeting either condition is reported as [`TrackStatus::Diverged`].
pub const CONVERGENCE_EPSILON: f32 = 0.01;

/// Why a feature point ended up where it did after tracking.
///
/// See [`TrackResult`] for the coordinate convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackStatus {
    /// The iteration converged inside the image; the position is trustworthy.
    Tracked,
    /// The search window left the image (in the previous or the next frame).
    OutOfBounds,
    /// The iteration hit `max_iterations` without converging (see
    /// [`CONVERGENCE_EPSILON`]), or a step exploded (non-finite or larger than
    /// the window). The reported position is the last estimate, not a lost
    /// track: callers that prefer OpenCV's semantics may still use it.
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
    let mut scratch = Scratch::default();
    let mut out = Vec::new();
    track_into(
        prev_pyramid,
        curr_pyramid,
        prev_points,
        predicted,
        window_size,
        max_iterations,
        min_eigen_threshold,
        &mut scratch,
        &mut out,
    );
    out
}

/// Reusable per-call scratch buffers for the Lucas-Kanade loop. Owned by
/// [`TrackerContext`] (or created transiently by the free functions) so the
/// steady-state hot path performs no heap allocation.
#[derive(Default)]
struct Scratch {
    prev_patch: Vec<f32>,
    ix_patch: Vec<f32>,
    iy_patch: Vec<f32>,
    displacements: Vec<(f32, f32)>,
    grad_x: Vec<i16>,
    grad_y: Vec<i16>,
}

impl Scratch {
    fn reserve(&mut self, capacity: &TrackerCapacity) {
        let patch = capacity.window_size * capacity.window_size;
        for buffer in [&mut self.prev_patch, &mut self.ix_patch, &mut self.iy_patch] {
            reserve_total(buffer, patch);
        }
        reserve_total(&mut self.displacements, capacity.points);
        let pixels = capacity.width as usize * capacity.height as usize;
        reserve_total(&mut self.grad_x, pixels);
        reserve_total(&mut self.grad_y, pixels);
    }
}

/// Core pyramidal Lucas-Kanade loop, writing one [`TrackResult`] per point into
/// `out`. All temporaries live in `scratch`; given sufficient capacity this is
/// allocation-free.
#[allow(clippy::too_many_arguments)]
fn track_into(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    predicted: Option<&[(f32, f32)]>,
    window_size: usize,
    max_iterations: usize,
    min_eigen_threshold: f32,
    scratch: &mut Scratch,
    out: &mut Vec<TrackResult>,
) {
    assert_eq!(prev_pyramid.len(), curr_pyramid.len());
    assert!(
        !prev_pyramid.is_empty(),
        "pyramid must have at least 1 level"
    );
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

    let Scratch {
        prev_patch,
        ix_patch,
        iy_patch,
        displacements,
        grad_x: grad_x_buf,
        grad_y: grad_y_buf,
    } = scratch;

    // Prepare reusable buffers. resize/clear+extend keep capacity, so none of
    // this allocates once the buffers are warm.
    prev_patch.resize(n_pixels, 0.0);
    ix_patch.resize(n_pixels, 0.0);
    iy_patch.resize(n_pixels, 0.0);

    // Total displacement per point, accumulated coarse-to-fine in level-0 units.
    // Seeding it from a prediction makes the coarsest level start at the
    // predicted position; everything else is identical to the zero-init path.
    displacements.clear();
    match predicted {
        Some(predicted) => displacements.extend(
            prev_points
                .iter()
                .zip(predicted.iter())
                .map(|((px, py), (gx, gy))| (gx - px, gy - py)),
        ),
        None => displacements.resize(prev_points.len(), (0.0, 0.0)),
    }

    // One shared gradient buffer sized to the largest (level-0) image; smaller
    // levels use a prefix slice.
    let (w0, h0) = prev_pyramid[0].dimensions();
    grad_x_buf.resize((w0 * h0) as usize, 0);
    grad_y_buf.resize((w0 * h0) as usize, 0);

    // Initialize results at the input positions; the loop refines them in place.
    out.clear();
    out.extend(prev_points.iter().map(|&(x, y)| TrackResult {
        pos: (x, y),
        status: TrackStatus::Tracked,
        error: f32::INFINITY,
    }));

    let kernel = Kernel::detect();

    // Process levels from top (coarse) to bottom (fine).
    for level in (0..n_levels).rev() {
        let prev_img = &prev_pyramid[level];
        let curr_img = &curr_pyramid[level];
        let (lw, lh) = prev_img.dimensions();
        let level_pixels = (lw * lh) as usize;

        compute_gradients_into(
            prev_img,
            &mut grad_x_buf[..level_pixels],
            &mut grad_y_buf[..level_pixels],
        );

        let level = Level {
            prev: prev_img.as_raw(),
            curr: curr_img.as_raw(),
            grad_x: &grad_x_buf[..level_pixels],
            grad_y: &grad_y_buf[..level_pixels],
            width: lw as usize,
            height: lh as usize,
            scale: 2f32.powi(level as i32),
            is_finest: level == 0,
            window_size,
            radius,
            max_iterations,
            min_eigen_threshold,
        };

        kernel.track_level(
            &level,
            prev_points,
            displacements,
            prev_patch,
            ix_patch,
            iy_patch,
            out,
        );
    }

    // Fold accumulated displacements into the reported positions.
    for (idx, (x, y)) in prev_points.iter().enumerate() {
        let (dx, dy) = displacements[idx];
        out[idx].pos = (x + dx, y + dy);
    }
}

/// Everything the per-level loop needs about one pyramid level.
struct Level<'a> {
    prev: &'a [u8],
    curr: &'a [u8],
    grad_x: &'a [i16],
    grad_y: &'a [i16],
    width: usize,
    height: usize,
    scale: f32,
    is_finest: bool,
    window_size: usize,
    radius: usize,
    max_iterations: usize,
    min_eigen_threshold: f32,
}

/// Instruction set the per-level loop is compiled for. The portable body is
/// written so the compiler can vectorize it for whatever the target enables
/// (SSE2 baseline, NEON, `simd128`); on x86 an AVX2 copy is selected at runtime.
#[derive(Clone, Copy)]
enum Kernel {
    Portable,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
}

impl Kernel {
    fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            return Kernel::Avx2;
        }
        Kernel::Portable
    }

    #[allow(clippy::too_many_arguments)]
    fn track_level(
        self,
        level: &Level,
        prev_points: &[(f32, f32)],
        displacements: &mut [(f32, f32)],
        prev_patch: &mut [f32],
        ix_patch: &mut [f32],
        iy_patch: &mut [f32],
        out: &mut [TrackResult],
    ) {
        match self {
            Kernel::Portable => track_level_impl(
                level,
                prev_points,
                displacements,
                prev_patch,
                ix_patch,
                iy_patch,
                out,
            ),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `detect` only returns `Avx2` after a positive runtime check.
            Kernel::Avx2 => unsafe {
                track_level_avx2(
                    level,
                    prev_points,
                    displacements,
                    prev_patch,
                    ix_patch,
                    iy_patch,
                    out,
                )
            },
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn track_level_avx2(
    level: &Level,
    prev_points: &[(f32, f32)],
    displacements: &mut [(f32, f32)],
    prev_patch: &mut [f32],
    ix_patch: &mut [f32],
    iy_patch: &mut [f32],
    out: &mut [TrackResult],
) {
    track_level_impl(
        level,
        prev_points,
        displacements,
        prev_patch,
        ix_patch,
        iy_patch,
        out,
    )
}

/// Refines every point's displacement at one pyramid level.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn track_level_impl(
    lv: &Level,
    prev_points: &[(f32, f32)],
    displacements: &mut [(f32, f32)],
    prev_patch: &mut [f32],
    ix_patch: &mut [f32],
    iy_patch: &mut [f32],
    out: &mut [TrackResult],
) {
    let n_pixels = lv.window_size * lv.window_size;
    let det_epsilon = 1e-6;
    let window_limit = lv.window_size as f32;

    for (idx, &(prev_x, prev_y)) in prev_points.iter().enumerate() {
        // Scale the original point for the current level.
        let x = prev_x / lv.scale;
        let y = prev_y / lv.scale;

        // Add the current displacement, scaled for this level.
        let mut dx = displacements[idx].0 / lv.scale;
        let mut dy = displacements[idx].1 / lv.scale;

        // The window must stay inside the previous image to build the patch.
        let Some(prev_win) = Window::new(x, y, lv.radius, lv.width, lv.height) else {
            out[idx].status = TrackStatus::OutOfBounds;
            continue;
        };

        // Spatial gradient matrix and cached previous/gradient patches.
        let (gxx, gxy, gyy) = build_patch(lv, &prev_win, prev_patch, ix_patch, iy_patch);

        // Reject low-texture windows up front (normalized by window area so
        // the threshold does not depend on `window_size`).
        let min_eig = min_eigenvalue(gxx, gxy, gyy) / n_pixels as f32;
        if min_eig < lv.min_eigen_threshold {
            out[idx].status = TrackStatus::LowTexture;
            if lv.is_finest {
                out[idx].error = window_error(lv, x + dx, y + dy, prev_patch);
            }
            continue;
        }

        let Some((inv_h00, inv_h01, inv_h11)) = invert_2x2(gxx, gxy, gyy, det_epsilon) else {
            out[idx].status = TrackStatus::LowTexture;
            continue;
        };

        // Refine the displacement at the current level.
        let mut converged = false;
        let mut out_of_bounds = false;
        let mut diverged = false;
        let mut prev_step = (0.0f32, 0.0f32);
        for iteration in 0..lv.max_iterations {
            let Some(win) = Window::new(x + dx, y + dy, lv.radius, lv.width, lv.height) else {
                out_of_bounds = true;
                break;
            };

            let (bx, by) = lk_step(lv, &win, prev_patch, ix_patch, iy_patch);

            let ddx = inv_h00 * bx + inv_h01 * by;
            let ddy = inv_h01 * bx + inv_h11 * by;
            dx += ddx;
            dy += ddy;

            // Guard against runaway steps.
            if !dx.is_finite()
                || !dy.is_finite()
                || ddx.abs() > window_limit
                || ddy.abs() > window_limit
            {
                diverged = true;
                break;
            }

            if ddx.abs() < CONVERGENCE_EPSILON && ddy.abs() < CONVERGENCE_EPSILON {
                converged = true;
                break;
            }

            // Two consecutive steps that cancel mean the estimate is bouncing
            // across the optimum; settle on the midpoint.
            if iteration > 0
                && (ddx + prev_step.0).abs() < CONVERGENCE_EPSILON
                && (ddy + prev_step.1).abs() < CONVERGENCE_EPSILON
            {
                dx -= ddx * 0.5;
                dy -= ddy * 0.5;
                converged = true;
                break;
            }
            prev_step = (ddx, ddy);
        }

        out[idx].status = if out_of_bounds {
            TrackStatus::OutOfBounds
        } else if diverged || !converged {
            TrackStatus::Diverged
        } else {
            TrackStatus::Tracked
        };

        // Update the total displacement with the current level scale.
        displacements[idx] = (dx * lv.scale, dy * lv.scale);

        if lv.is_finest {
            out[idx].error = if out_of_bounds {
                f32::INFINITY
            } else {
                window_error(lv, x + dx, y + dy, prev_patch)
            };
        }
    }
}

/// A sampling window anchored at an integer top-left corner. Because the
/// window offsets are integers, every sample shares the same fractional
/// position and therefore the same four bilinear weights; each window row is
/// then a contiguous run of `window_size + 1` pixels from two image rows.
struct Window {
    x0: usize,
    y0: usize,
    w00: f32,
    w10: f32,
    w01: f32,
    w11: f32,
}

impl Window {
    /// Returns `None` when the full 2x2 bilinear footprint of the window around
    /// `(cx, cy)` is not inside a `width` x `height` image.
    #[inline(always)]
    fn new(cx: f32, cy: f32, radius: usize, width: usize, height: usize) -> Option<Window> {
        let r = radius as f32;
        // Written so that NaN coordinates fail the test.
        if !(cx >= r && cy >= r && cx < width as f32 - r - 1.0 && cy < height as f32 - r - 1.0) {
            return None;
        }
        let fx0 = cx.floor();
        let fy0 = cy.floor();
        let fx = cx - fx0;
        let fy = cy - fy0;
        Some(Window {
            x0: fx0 as usize - radius,
            y0: fy0 as usize - radius,
            w00: (1.0 - fx) * (1.0 - fy),
            w10: fx * (1.0 - fy),
            w01: (1.0 - fx) * fy,
            w11: fx * fy,
        })
    }

    /// The same window with every weight multiplied by `factor`.
    #[inline(always)]
    fn scaled(&self, factor: f32) -> Window {
        Window {
            x0: self.x0,
            y0: self.y0,
            w00: self.w00 * factor,
            w10: self.w10 * factor,
            w01: self.w01 * factor,
            w11: self.w11 * factor,
        }
    }

    /// The two image rows feeding window row `j`: `size + 1` pixels each.
    #[inline(always)]
    fn rows<'a, T>(
        &self,
        data: &'a [T],
        width: usize,
        j: usize,
        size: usize,
    ) -> (&'a [T], &'a [T]) {
        let start = (self.y0 + j) * width + self.x0;
        (
            &data[start..start + size + 1],
            &data[start + width..start + width + size + 1],
        )
    }
}

/// Bilinear samples of `L` consecutive window pixels from two source rows.
#[inline(always)]
fn sample_lanes<T: Copy + Into<f32>, const L: usize>(win: &Window, a: &[T], b: &[T]) -> [f32; L] {
    let a0: &[T; L] = a[..L].try_into().unwrap();
    let a1: &[T; L] = a[1..=L].try_into().unwrap();
    let b0: &[T; L] = b[..L].try_into().unwrap();
    let b1: &[T; L] = b[1..=L].try_into().unwrap();
    let mut s = [0f32; L];
    for k in 0..L {
        s[k] = win.w00 * a0[k].into()
            + win.w10 * a1[k].into()
            + win.w01 * b0[k].into()
            + win.w11 * b1[k].into();
    }
    s
}

#[inline(always)]
fn total<const L: usize>(acc: &[f32; L]) -> f32 {
    acc.iter().sum()
}

/// Fills the previous-image and gradient patches for `win` and returns the
/// spatial gradient matrix `(gxx, gxy, gyy)`.
#[inline(always)]
fn build_patch(
    lv: &Level,
    win: &Window,
    prev_patch: &mut [f32],
    ix_patch: &mut [f32],
    iy_patch: &mut [f32],
) -> (f32, f32, f32) {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn lanes<const L: usize>(
        win: &Window,
        gwin: &Window,
        (pa, pb): (&[u8], &[u8]),
        (xa, xb): (&[i16], &[i16]),
        (ya, yb): (&[i16], &[i16]),
        p: &mut [f32],
        ox: &mut [f32],
        oy: &mut [f32],
        acc: &mut [[f32; L]; 3],
    ) {
        let s = sample_lanes::<u8, L>(win, pa, pb);
        let sx = sample_lanes::<i16, L>(gwin, xa, xb);
        let sy = sample_lanes::<i16, L>(gwin, ya, yb);
        let p: &mut [f32; L] = (&mut p[..L]).try_into().unwrap();
        let ox: &mut [f32; L] = (&mut ox[..L]).try_into().unwrap();
        let oy: &mut [f32; L] = (&mut oy[..L]).try_into().unwrap();
        *p = s;
        *ox = sx;
        *oy = sy;
        for k in 0..L {
            acc[0][k] += sx[k] * sx[k];
            acc[1][k] += sx[k] * sy[k];
            acc[2][k] += sy[k] * sy[k];
        }
    }

    let size = lv.window_size;
    // Scharr outputs are 32x the intensity slope.
    let gwin = win.scaled(1.0 / 32.0);
    let mut acc8 = [[0f32; 8]; 3];
    let mut acc4 = [[0f32; 4]; 3];
    let mut acc1 = [[0f32; 1]; 3];

    for j in 0..size {
        let (pa, pb) = win.rows(lv.prev, lv.width, j, size);
        let (xa, xb) = win.rows(lv.grad_x, lv.width, j, size);
        let (ya, yb) = win.rows(lv.grad_y, lv.width, j, size);
        let p = &mut prev_patch[j * size..][..size];
        let ox = &mut ix_patch[j * size..][..size];
        let oy = &mut iy_patch[j * size..][..size];

        let mut i = 0;
        while i + 8 <= size {
            lanes::<8>(
                win,
                &gwin,
                (&pa[i..], &pb[i..]),
                (&xa[i..], &xb[i..]),
                (&ya[i..], &yb[i..]),
                &mut p[i..],
                &mut ox[i..],
                &mut oy[i..],
                &mut acc8,
            );
            i += 8;
        }
        if i + 4 <= size {
            lanes::<4>(
                win,
                &gwin,
                (&pa[i..], &pb[i..]),
                (&xa[i..], &xb[i..]),
                (&ya[i..], &yb[i..]),
                &mut p[i..],
                &mut ox[i..],
                &mut oy[i..],
                &mut acc4,
            );
            i += 4;
        }
        while i < size {
            lanes::<1>(
                win,
                &gwin,
                (&pa[i..], &pb[i..]),
                (&xa[i..], &xb[i..]),
                (&ya[i..], &yb[i..]),
                &mut p[i..],
                &mut ox[i..],
                &mut oy[i..],
                &mut acc1,
            );
            i += 1;
        }
    }

    (
        total(&acc8[0]) + total(&acc4[0]) + total(&acc1[0]),
        total(&acc8[1]) + total(&acc4[1]) + total(&acc1[1]),
        total(&acc8[2]) + total(&acc4[2]) + total(&acc1[2]),
    )
}

/// One Lucas-Kanade update: the gradient-weighted residual `(bx, by)` between
/// the cached previous patch and the next image sampled at `win`.
#[inline(always)]
fn lk_step(
    lv: &Level,
    win: &Window,
    prev_patch: &[f32],
    ix_patch: &[f32],
    iy_patch: &[f32],
) -> (f32, f32) {
    #[inline(always)]
    fn lanes<const L: usize>(
        win: &Window,
        (a, b): (&[u8], &[u8]),
        p: &[f32],
        gx: &[f32],
        gy: &[f32],
        acc: &mut [[f32; L]; 2],
    ) {
        let s = sample_lanes::<u8, L>(win, a, b);
        let p: &[f32; L] = p[..L].try_into().unwrap();
        let gx: &[f32; L] = gx[..L].try_into().unwrap();
        let gy: &[f32; L] = gy[..L].try_into().unwrap();
        for k in 0..L {
            let e = p[k] - s[k];
            acc[0][k] += gx[k] * e;
            acc[1][k] += gy[k] * e;
        }
    }

    let size = lv.window_size;
    let mut acc8 = [[0f32; 8]; 2];
    let mut acc4 = [[0f32; 4]; 2];
    let mut acc1 = [[0f32; 1]; 2];

    for j in 0..size {
        let (a, b) = win.rows(lv.curr, lv.width, j, size);
        let p = &prev_patch[j * size..][..size];
        let gx = &ix_patch[j * size..][..size];
        let gy = &iy_patch[j * size..][..size];

        let mut i = 0;
        while i + 8 <= size {
            lanes::<8>(
                win,
                (&a[i..], &b[i..]),
                &p[i..],
                &gx[i..],
                &gy[i..],
                &mut acc8,
            );
            i += 8;
        }
        if i + 4 <= size {
            lanes::<4>(
                win,
                (&a[i..], &b[i..]),
                &p[i..],
                &gx[i..],
                &gy[i..],
                &mut acc4,
            );
            i += 4;
        }
        while i < size {
            lanes::<1>(
                win,
                (&a[i..], &b[i..]),
                &p[i..],
                &gx[i..],
                &gy[i..],
                &mut acc1,
            );
            i += 1;
        }
    }

    (
        total(&acc8[0]) + total(&acc4[0]) + total(&acc1[0]),
        total(&acc8[1]) + total(&acc4[1]) + total(&acc1[1]),
    )
}

/// Mean absolute photometric residual between the cached previous patch and the
/// next image sampled around `(cx, cy)`. Returns [`f32::INFINITY`] if the window
/// is out of bounds.
#[inline(always)]
fn window_error(lv: &Level, cx: f32, cy: f32, prev_patch: &[f32]) -> f32 {
    #[inline(always)]
    fn lanes<const L: usize>(win: &Window, (a, b): (&[u8], &[u8]), p: &[f32], acc: &mut [f32; L]) {
        let s = sample_lanes::<u8, L>(win, a, b);
        let p: &[f32; L] = p[..L].try_into().unwrap();
        for k in 0..L {
            acc[k] += (p[k] - s[k]).abs();
        }
    }

    let Some(win) = Window::new(cx, cy, lv.radius, lv.width, lv.height) else {
        return f32::INFINITY;
    };
    let size = lv.window_size;
    let mut acc8 = [0f32; 8];
    let mut acc4 = [0f32; 4];
    let mut acc1 = [0f32; 1];

    for j in 0..size {
        let (a, b) = win.rows(lv.curr, lv.width, j, size);
        let p = &prev_patch[j * size..][..size];
        let mut i = 0;
        while i + 8 <= size {
            lanes::<8>(&win, (&a[i..], &b[i..]), &p[i..], &mut acc8);
            i += 8;
        }
        if i + 4 <= size {
            lanes::<4>(&win, (&a[i..], &b[i..]), &p[i..], &mut acc4);
            i += 4;
        }
        while i < size {
            lanes::<1>(&win, (&a[i..], &b[i..]), &p[i..], &mut acc1);
            i += 1;
        }
    }

    (total(&acc8) + total(&acc4) + total(&acc1)) / (size * size) as f32
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
    let mut scratch = Scratch::default();
    let mut forward = Vec::new();
    track_into(
        prev_pyramid,
        next_pyramid,
        prev_points,
        predicted,
        window_size,
        max_iterations,
        min_eigen_threshold,
        &mut scratch,
        &mut forward,
    );

    let forward_pos: Vec<(f32, f32)> = forward.iter().map(|r| r.pos).collect();
    let mut backward = Vec::new();
    // Seed the backward pass at the original points (the round-trip is expected
    // to return there). This keeps the check robust under large motion, as in
    // OpenCV's OPTFLOW_USE_INITIAL_FLOW reverse check, without weakening it: a
    // genuinely wrong forward match still fails to land back within threshold.
    track_into(
        next_pyramid,
        prev_pyramid,
        &forward_pos,
        Some(prev_points),
        window_size,
        max_iterations,
        min_eigen_threshold,
        &mut scratch,
        &mut backward,
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

fn invert_2x2(a00: f32, a01: f32, a11: f32, det_epsilon: f32) -> Option<(f32, f32, f32)> {
    let det = a00 * a11 - a01 * a01;
    if det.abs() <= det_epsilon {
        return None;
    }

    let inv_det = 1.0 / det;
    Some((a11 * inv_det, -a01 * inv_det, a00 * inv_det))
}

/// Upper bounds for the work a tracker context will see, used to allocate
/// every buffer up front with [`TrackerContext::with_capacity`] or
/// [`TrackerContext::reserve`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackerCapacity {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pyramid levels passed to `prepare`.
    pub levels: usize,
    /// Largest `window_size` passed to `track` / `track_fb`.
    pub window_size: usize,
    /// Most points per `track` / `track_fb` call.
    pub points: usize,
}

/// Reusable owner of every buffer the tracking hot path touches: both frame
/// pyramids, the Lucas-Kanade scratch, the result vector and the
/// forward-backward intermediates.
///
/// Create one per tracking thread and call [`prepare`](Self::prepare) then
/// [`track`](Self::track) / [`track_fb`](Self::track_fb) each frame. All
/// buffers are resized in place and reused. To keep allocation out of the
/// per-frame path, create the context with
/// [`with_capacity`](Self::with_capacity): frames of the reserved size and
/// level count, with at most the reserved window size and point count, then
/// perform **no heap allocation**. Without a reservation the first frame
/// allocates, and so does any later call with more points than before. The
/// free functions ([`calc_optical_flow_ex`], [`calc_optical_flow_fb`]) are
/// thin convenience wrappers that allocate their own scratch.
#[derive(Default)]
pub struct TrackerContext {
    prev_pyramid: Vec<GrayImage>,
    next_pyramid: Vec<GrayImage>,
    scratch: Scratch,
    results: Vec<TrackResult>,
    forward_pos: Vec<(f32, f32)>,
    backward: Vec<TrackResult>,
}

impl TrackerContext {
    /// Creates an empty context. Buffers grow to fit on the first
    /// [`prepare`](Self::prepare) / [`track`](Self::track) call and are reused
    /// thereafter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a context with every buffer allocated for `capacity`.
    pub fn with_capacity(capacity: TrackerCapacity) -> Self {
        let mut context = Self::default();
        context.reserve(capacity);
        context
    }

    /// Allocates every buffer for `capacity` now, so that later calls within
    /// it do not allocate. Existing capacity is kept.
    pub fn reserve(&mut self, capacity: TrackerCapacity) {
        let TrackerCapacity {
            width,
            height,
            levels,
            points,
            ..
        } = capacity;
        reserve_pyramid(&mut self.prev_pyramid, width, height, levels);
        reserve_pyramid(&mut self.next_pyramid, width, height, levels);
        self.scratch.reserve(&capacity);
        reserve_total(&mut self.results, points);
        reserve_total(&mut self.forward_pos, points);
        reserve_total(&mut self.backward, points);
    }

    /// Builds the previous- and next-frame pyramids into the context's reusable
    /// buffers. Zero-alloc in steady state (same image size and `levels`).
    pub fn prepare(&mut self, prev: &GrayImage, next: &GrayImage, levels: usize) {
        build_pyramid_into(prev, levels, &mut self.prev_pyramid);
        build_pyramid_into(next, levels, &mut self.next_pyramid);
    }

    /// The previous-frame pyramid built by the last [`prepare`](Self::prepare).
    pub fn prev_pyramid(&self) -> &[GrayImage] {
        &self.prev_pyramid
    }

    /// The next-frame pyramid built by the last [`prepare`](Self::prepare).
    pub fn next_pyramid(&self) -> &[GrayImage] {
        &self.next_pyramid
    }

    /// Tracks `prev_points` using the prepared pyramids, returning the results
    /// held inside the context. See [`calc_optical_flow_ex`] for the argument
    /// semantics. Allocation-free in steady state.
    pub fn track(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        window_size: usize,
        max_iterations: usize,
        min_eigen_threshold: f32,
    ) -> &[TrackResult] {
        track_into(
            &self.prev_pyramid,
            &self.next_pyramid,
            prev_points,
            predicted,
            window_size,
            max_iterations,
            min_eigen_threshold,
            &mut self.scratch,
            &mut self.results,
        );
        &self.results
    }

    /// Forward-backward consistent tracking using the prepared pyramids. See
    /// [`calc_optical_flow_fb`] for semantics. Reuses the context's scratch and
    /// intermediate point buffers, so it is allocation-free in steady state.
    pub fn track_fb(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        window_size: usize,
        max_iterations: usize,
        min_eigen_threshold: f32,
        fb_threshold: f32,
    ) -> &[TrackResult] {
        track_into(
            &self.prev_pyramid,
            &self.next_pyramid,
            prev_points,
            predicted,
            window_size,
            max_iterations,
            min_eigen_threshold,
            &mut self.scratch,
            &mut self.results,
        );

        self.forward_pos.clear();
        self.forward_pos.extend(self.results.iter().map(|r| r.pos));

        // Seed the backward pass at the original points (see calc_optical_flow_fb).
        track_into(
            &self.next_pyramid,
            &self.prev_pyramid,
            &self.forward_pos,
            Some(prev_points),
            window_size,
            max_iterations,
            min_eigen_threshold,
            &mut self.scratch,
            &mut self.backward,
        );

        mark_fb_inconsistent(&mut self.results, &self.backward, prev_points, fb_threshold);
        &self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

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

    /// Reference bilinear sample with the crate's pixel-center convention.
    fn bilinear(img: &GrayImage, x: f32, y: f32) -> f32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let px = |dx: u32, dy: u32| img.get_pixel(x0 as u32 + dx, y0 as u32 + dy)[0] as f32;
        px(0, 0) * (1.0 - fx) * (1.0 - fy)
            + px(1, 0) * fx * (1.0 - fy)
            + px(0, 1) * (1.0 - fx) * fy
            + px(1, 1) * fx * fy
    }

    #[test]
    fn window_sampling_matches_per_pixel_bilinear_reference() {
        let img = GrayImage::from_fn(40, 30, |x, y| Luma([((x * 37 + y * 11) ^ (x * y)) as u8]));
        let mut grad_x = vec![0i16; 40 * 30];
        let mut grad_y = vec![0i16; 40 * 30];
        compute_gradients_into(&img, &mut grad_x, &mut grad_y);

        for &size in &[1usize, 5, 7, 11, 21] {
            let radius = size / 2;
            let lv = Level {
                prev: img.as_raw(),
                curr: img.as_raw(),
                grad_x: &grad_x,
                grad_y: &grad_y,
                width: 40,
                height: 30,
                scale: 1.0,
                is_finest: true,
                window_size: size,
                radius,
                max_iterations: 1,
                min_eigen_threshold: 0.0,
            };
            let (cx, cy) = (17.3f32, 14.6f32);
            let win = Window::new(cx, cy, radius, 40, 30).unwrap();
            let mut patch = vec![0f32; size * size];
            let mut ix = vec![0f32; size * size];
            let mut iy = vec![0f32; size * size];
            build_patch(&lv, &win, &mut patch, &mut ix, &mut iy);

            for j in 0..size {
                for i in 0..size {
                    let (sx, sy) = (
                        cx + (i as f32 - radius as f32),
                        cy + (j as f32 - radius as f32),
                    );
                    let expected = bilinear(&img, sx, sy);
                    let got = patch[j * size + i];
                    assert!(
                        (got - expected).abs() < 1e-3,
                        "size {size} at ({i},{j}): {got} vs {expected}"
                    );
                }
            }
            // The next image is the previous image, so the residual at the same
            // position is zero and a step away from it is not.
            assert!(window_error(&lv, cx, cy, &patch) < 1e-4);
            let (bx, by) = lk_step(&lv, &win, &patch, &ix, &iy);
            assert!(
                bx.abs() < 1e-2 && by.abs() < 1e-2,
                "size {size}: ({bx}, {by})"
            );
        }
    }

    #[test]
    fn window_rejects_footprints_touching_the_border() {
        assert!(Window::new(10.0, 10.0, 10, 40, 40).is_some());
        assert!(Window::new(9.9, 10.0, 10, 40, 40).is_none());
        assert!(Window::new(28.9, 10.0, 10, 40, 40).is_some());
        assert!(Window::new(29.0, 10.0, 10, 40, 40).is_none());
        assert!(Window::new(f32::NAN, 10.0, 10, 40, 40).is_none());
    }
}
