use image::{GrayImage, ImageBuffer, Luma, Pixel};

use crate::features::reserve_total;
use crate::float_image::FloatImageView;
use crate::pyramid::{
    build_float_pyramid_into, build_pyramid_into, level_dimensions, pyramid_level_count,
    reserve_pyramid,
};
use crate::utils::fast_gradients::compute_gradients_into;
use crate::utils::float_gradients::scharr_f32;

mod fixed;

/// One level of an f32 image pyramid, as built by [`FloatTrackerContext`].
pub type FloatPyramidLevel = ImageBuffer<Luma<f32>, Vec<f32>>;

/// Default [`TrackParams::min_eigen_threshold`] of the 8-bit tracker.
///
/// The minimum eigenvalue of the per-window spatial gradient matrix is
/// normalized by the window area before comparison, so this threshold is
/// independent of `window_size`. Its unit is (grey levels / pixel)^2. A window
/// flatter than this is reported as [`TrackStatus::LowTexture`]. The f32
/// tracker has no default: its threshold is in the image's own unit.
///
/// Thresholds below about 1e-4 are not meaningful: at the strongest 8-bit
/// edges, f32 rounding of the eigenvalue alone reaches about that size, so
/// such edges could pass as trackable windows at random.
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

/// Lucas-Kanade parameters of a tracking call.
///
/// The fields are named so that they cannot be swapped unnoticed. For 8-bit
/// images, for example:
///
/// ```
/// use optical_flow_lk::{DEFAULT_MIN_EIGEN_THRESHOLD, TrackParams};
///
/// let params = TrackParams {
///     window_size: 21,
///     max_iterations: 30,
///     min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackParams {
    /// Side of the square search window in pixels; must be odd.
    pub window_size: usize,
    /// Most iterations per pyramid level, see [`CONVERGENCE_EPSILON`].
    pub max_iterations: usize,
    /// Windows whose minimum gradient eigenvalue, normalized by the window
    /// area, is below this value are reported as [`TrackStatus::LowTexture`].
    /// In (grey levels / pixel)^2 for 8-bit images, see
    /// [`DEFAULT_MIN_EIGEN_THRESHOLD`]; in (input unit / pixel)^2 for the f32
    /// tracker, which has no default.
    pub min_eigen_threshold: f32,
}

/// Arithmetic of the 8-bit tracker's window kernels, see
/// [`TrackOptions::kernel`].
///
/// The fixed-point kernels follow the scheme of OpenCV's
/// `calcOpticalFlowPyrLK`: window samples are bilinear interpolations with
/// 14-bit integer weights, kept as 16-bit integers with 5 fractional bits for
/// intensities and 6 for gradients, and every window sum is an exact integer
/// accumulated from 16-bit products. The integers are the same on every
/// instruction set, so results are bit for bit identical on x86 (SSE2,
/// AVX2), wasm32 (`simd128`) and the scalar fallback. Against the f32
/// reference, tracked positions typically differ by a few ten-thousandths of
/// a pixel; see the changelog of 0.6.0 for measurements.
///
/// The f32 tracker ([`FloatTrackerContext`]) always uses its f32 kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TrackKernel {
    /// Fixed point where the target has SIMD kernels for it: x86 and x86-64
    /// with SSE2 (AVX2 when the CPU has it) and wasm32 built with `simd128`.
    /// The f32 reference elsewhere, including aarch64, which has no
    /// fixed-point SIMD kernel yet.
    #[default]
    Auto,
    /// Fixed point on every target, with a scalar kernel where there is no
    /// SIMD one. Gives the same results as `Auto` on SIMD targets.
    FixedPoint,
    /// The f32 reference kernel, the arithmetic of 0.5 and earlier releases.
    /// Windows wider than 127 px always use it.
    Reference,
}

/// Options of a tracking call beyond [`TrackParams`], for
/// [`TrackerContext::track_with`] and [`TrackerContext::track_fb_with`] and
/// their [`FloatTrackerContext`] counterparts. The default reproduces
/// `track` and `track_fb`.
///
/// ```
/// use optical_flow_lk::TrackOptions;
///
/// // Check the round trip on the two finest levels only.
/// const OPTIONS: TrackOptions = TrackOptions::new().with_fb_levels(2);
/// assert_eq!(OPTIONS.fb_levels, Some(2));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TrackOptions {
    /// Pyramid level the forward pass starts at, 0 being full resolution.
    /// `None` starts at the coarsest level; levels above it are clamped to
    /// it. The displacement from `predicted` (or zero) seeds the start level.
    ///
    /// Each level above full resolution doubles the motion the iteration can
    /// recover from its seed, about half a window at level 0. Starting lower
    /// saves the coarse levels' work when a prediction (a gyroscope, a
    /// constant-velocity model) is already that close, and loses the capture
    /// range they give otherwise.
    pub start_level: Option<usize>,
    /// Pyramid levels the backward pass of `track_fb_with` runs, the finest
    /// ones: `Some(k)` runs levels `k - 1` down to 0, `None` all of them, as
    /// `track_fb` does. Must be at least 1; ignored by `track_with`.
    ///
    /// The backward pass tracks the forward results back into the previous
    /// frame, seeded at the original points, which a consistent round trip
    /// returns to. A correct forward match therefore needs no coarse levels
    /// to come back. Fewer levels make the check cheaper and slightly more
    /// permissive: a wrong match that the coarse levels would have carried
    /// further away can stay within the threshold. On the bundled example
    /// pair, see the changelog of 0.6.0 for how many points change status.
    pub fb_levels: Option<usize>,
    /// Arithmetic of the 8-bit tracker. Ignored by [`FloatTrackerContext`].
    pub kernel: TrackKernel,
}

impl TrackOptions {
    /// The default options: all levels in both passes, [`TrackKernel::Auto`].
    pub const fn new() -> Self {
        TrackOptions {
            start_level: None,
            fb_levels: None,
            kernel: TrackKernel::Auto,
        }
    }

    /// Starts the forward pass at pyramid level `level`, see
    /// [`start_level`](Self::start_level).
    pub const fn with_start_level(mut self, level: usize) -> Self {
        self.start_level = Some(level);
        self
    }

    /// Runs the backward pass over the finest `levels` levels, see
    /// [`fb_levels`](Self::fb_levels).
    pub const fn with_fb_levels(mut self, levels: usize) -> Self {
        self.fb_levels = Some(levels);
        self
    }

    /// Selects the arithmetic of the 8-bit tracker.
    pub const fn with_kernel(mut self, kernel: TrackKernel) -> Self {
        self.kernel = kernel;
        self
    }
}

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
    /// position, in the input's intensity unit: grey levels for 8-bit images,
    /// the image's own unit (e.g. °C) for the f32 tracker.
    ///
    /// Comparable across points regardless of `window_size`; intended for
    /// downstream outlier gating. It is [`f32::INFINITY`] when no residual
    /// could be measured (the window was out of bounds).
    pub error: f32,
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
/// * `params` - window size, iterations and texture threshold, see
///   [`TrackParams`]
///
/// # Panics
/// Panics if the pyramids differ in level count or level sizes, if
/// `params.window_size` is even, or if `predicted` is `Some` but its length
/// differs from `prev_points`.
///
/// # Returns
/// One [`TrackResult`] per input point, in the same order.
pub fn calc_optical_flow_ex(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    predicted: Option<&[(f32, f32)]>,
    params: TrackParams,
) -> Vec<TrackResult> {
    let mut scratch = Scratch::default();
    let mut out = Vec::new();
    track_into(
        prev_pyramid,
        &mut Gradients::default(),
        curr_pyramid,
        prev_points,
        predicted,
        params,
        None,
        GrayImage::select(TrackKernel::Auto, params.window_size),
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
    /// Patches of the f32 kernels: intensities and both gradients.
    float_patch: [Vec<f32>; 3],
    /// Patches of the fixed-point kernels, rows padded to whole lane chunks.
    fixed_patch: [Vec<i16>; 3],
    displacements: Vec<(f32, f32)>,
}

impl Scratch {
    fn reserve(&mut self, capacity: &TrackerCapacity) {
        let size = capacity.window_size;
        for buffer in &mut self.float_patch {
            reserve_total(buffer, size * size);
        }
        for buffer in &mut self.fixed_patch {
            reserve_total(buffer, size * fixed::stride(size));
        }
        reserve_total(&mut self.displacements, capacity.points);
    }
}

/// Scharr gradients of the levels of one pyramid, each computed when first
/// needed and kept until the pyramid changes. The contexts keep one per
/// pyramid, so every frame's gradients are computed once, however many
/// passes read them; the free functions use a transient one per call.
#[derive(Default)]
struct Gradients<G> {
    levels: Vec<GradientLevel<G>>,
}

#[derive(Default)]
struct GradientLevel<G> {
    x: Vec<G>,
    y: Vec<G>,
    /// Whether `x` and `y` hold the gradients of the pyramid's current level.
    ready: bool,
}

impl<G: Copy + Default> Gradients<G> {
    /// Forgets every level, for a pyramid that was rebuilt.
    fn invalidate(&mut self) {
        for level in &mut self.levels {
            level.ready = false;
        }
    }

    /// Allocates the planes of a `width` x `height` pyramid with `levels`
    /// levels.
    fn reserve(&mut self, width: u32, height: u32, levels: usize) {
        for (index, (w, h)) in level_dimensions(width, height, levels).enumerate() {
            if index == self.levels.len() {
                self.levels.push(GradientLevel::default());
            }
            let pixels = w as usize * h as usize;
            reserve_total(&mut self.levels[index].x, pixels);
            reserve_total(&mut self.levels[index].y, pixels);
        }
    }

    /// The gradients of `pyramid[level]`, computed now unless they are kept
    /// from an earlier call.
    fn level<I: TrackImage<Gradient = G>>(&mut self, pyramid: &[I], level: usize) -> (&[G], &[G]) {
        if self.levels.len() < pyramid.len() {
            self.levels
                .resize_with(pyramid.len(), GradientLevel::default);
        }
        let image = &pyramid[level];
        let slot = &mut self.levels[level];
        if !slot.ready {
            let (w, h) = image.dimensions();
            let pixels = w as usize * h as usize;
            slot.x.resize(pixels, G::default());
            slot.y.resize(pixels, G::default());
            image.gradients_into(&mut slot.x, &mut slot.y);
            slot.ready = true;
        }
        (&slot.x, &slot.y)
    }
}

/// Core pyramidal Lucas-Kanade loop, writing one [`TrackResult`] per point into
/// `out`. `prev_gradients` caches the gradients of `prev_pyramid`. The pass
/// runs from `start_level` (`None`: the top of the pyramid) down to full
/// resolution with `kernel`. All temporaries live in `scratch`; given
/// sufficient capacity this is allocation-free.
#[allow(clippy::too_many_arguments)]
fn track_into<I: TrackImage>(
    prev_pyramid: &[I],
    prev_gradients: &mut Gradients<I::Gradient>,
    curr_pyramid: &[I],
    prev_points: &[(f32, f32)],
    predicted: Option<&[(f32, f32)]>,
    params: TrackParams,
    start_level: Option<usize>,
    kernel: I::Kernel,
    scratch: &mut Scratch,
    out: &mut Vec<TrackResult>,
) {
    assert_eq!(prev_pyramid.len(), curr_pyramid.len());
    assert!(
        !prev_pyramid.is_empty(),
        "pyramid must have at least 1 level"
    );
    assert!(
        prev_pyramid
            .iter()
            .zip(curr_pyramid)
            .all(|(prev, curr)| prev.dimensions() == curr.dimensions()),
        "both pyramids must have the same level sizes"
    );
    let TrackParams {
        window_size,
        max_iterations,
        min_eigen_threshold,
    } = params;
    assert!(window_size % 2 == 1, "Window size must be odd");
    if let Some(predicted) = predicted {
        assert_eq!(
            predicted.len(),
            prev_points.len(),
            "predicted must have one entry per prev_point"
        );
    }

    let top = prev_pyramid.len() - 1;
    let start = start_level.map_or(top, |level| level.min(top));
    let radius = window_size / 2;

    // Total displacement per point, accumulated coarse-to-fine in level-0 units.
    // Seeding it from a prediction makes the start level begin at the
    // predicted position; everything else is identical to the zero-init path.
    let displacements = &mut scratch.displacements;
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

    // Initialize results at the input positions; the loop refines them in place.
    out.clear();
    out.extend(prev_points.iter().map(|&(x, y)| TrackResult {
        pos: (x, y),
        status: TrackStatus::Tracked,
        error: f32::INFINITY,
    }));

    // Process levels from the start level (coarse) to the bottom (fine).
    for level in (0..=start).rev() {
        let prev_img = &prev_pyramid[level];
        let curr_img = &curr_pyramid[level];
        let (lw, lh) = prev_img.dimensions();
        let (grad_x, grad_y) = prev_gradients.level(prev_pyramid, level);

        let level = Level::<I> {
            prev: prev_img.pixels(),
            curr: curr_img.pixels(),
            grad_x,
            grad_y,
            width: lw as usize,
            height: lh as usize,
            scale: 2f32.powi(level as i32),
            is_finest: level == 0,
            window_size,
            radius,
            max_iterations,
            min_eigen_threshold,
        };

        I::track_level(kernel, &level, prev_points, scratch, out);
    }

    // Fold accumulated displacements into the reported positions.
    for (idx, (x, y)) in prev_points.iter().enumerate() {
        let (dx, dy) = scratch.displacements[idx];
        out[idx].pos = (x + dx, y + dy);
    }
}

/// Everything the per-level loop needs about one pyramid level.
struct Level<'a, I: TrackImage> {
    prev: &'a [I::Pixel],
    curr: &'a [I::Pixel],
    grad_x: &'a [I::Gradient],
    grad_y: &'a [I::Gradient],
    width: usize,
    height: usize,
    scale: f32,
    is_finest: bool,
    window_size: usize,
    radius: usize,
    max_iterations: usize,
    min_eigen_threshold: f32,
}

/// The f32 reference kernel, also the only kernel of the f32 tracker. The
/// portable body is written so the compiler can vectorize it for whatever the
/// target enables (SSE2 baseline, NEON, `simd128`); on x86 an AVX2 copy is
/// selected at runtime.
#[derive(Clone, Copy, Debug)]
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

    /// Refines every point's displacement at one level.
    fn track_level<I: TrackImage>(
        self,
        level: &Level<I>,
        prev_points: &[(f32, f32)],
        scratch: &mut Scratch,
        out: &mut [TrackResult],
    ) {
        let n_pixels = level.window_size * level.window_size;
        let Scratch {
            float_patch: [prev, ix, iy],
            displacements,
            ..
        } = scratch;
        for patch in [&mut *prev, &mut *ix, &mut *iy] {
            patch.resize(n_pixels, 0.0);
        }
        let windows = FloatWindows { prev, ix, iy };
        match self {
            Kernel::Portable => track_level_impl(level, prev_points, displacements, windows, out),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `detect` only returns `Avx2` after a positive runtime check.
            Kernel::Avx2 => unsafe {
                track_level_avx2(level, prev_points, displacements, windows, out)
            },
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn track_level_avx2<I: TrackImage, W: Windows<I>>(
    level: &Level<I>,
    prev_points: &[(f32, f32)],
    displacements: &mut [(f32, f32)],
    windows: W,
    out: &mut [TrackResult],
) {
    track_level_impl(level, prev_points, displacements, windows, out)
}

/// A fixed-point kernel of the 8-bit tracker, see the `fixed` module. All of
/// them return the same results. aarch64 has no NEON kernel yet and runs the
/// scalar one when `TrackKernel::FixedPoint` asks for fixed point.
#[derive(Clone, Copy, Debug)]
enum FixedKernel {
    /// The kernel of targets without SIMD kernels; the tests run it on all.
    #[cfg_attr(
        any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sse2"
            ),
            all(target_arch = "wasm32", target_feature = "simd128")
        ),
        allow(dead_code)
    )]
    Scalar,
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(fixed::Avx2),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

impl FixedKernel {
    /// The SIMD kernel every CPU of the compilation target has.
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))]
    const BASELINE: Self = FixedKernel::Sse2;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    const BASELINE: Self = FixedKernel::Simd128;
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    const BASELINE: Self = FixedKernel::Scalar;

    /// The fastest fixed-point kernel of this CPU; `Scalar` without SIMD.
    fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(avx2) = fixed::Avx2::detect() {
            return FixedKernel::Avx2(avx2);
        }
        Self::BASELINE
    }

    /// Refines every point's displacement at one level.
    fn track_level(
        self,
        level: &Level<GrayImage>,
        prev_points: &[(f32, f32)],
        scratch: &mut Scratch,
        out: &mut [TrackResult],
    ) {
        let size = level.window_size;
        let stride = fixed::stride(size);
        let Scratch {
            fixed_patch: [pixels, grad_x, grad_y],
            displacements,
            ..
        } = scratch;
        for patch in [&mut *pixels, &mut *grad_x, &mut *grad_y] {
            patch.resize(size * stride, 0);
        }
        let patch = fixed::Patch {
            size,
            stride,
            pixels,
            grad_x,
            grad_y,
        };
        fn run<K: fixed::Ops>(
            level: &Level<GrayImage>,
            prev_points: &[(f32, f32)],
            displacements: &mut [(f32, f32)],
            patch: fixed::Patch<'_>,
            ops: K,
            out: &mut [TrackResult],
        ) {
            let windows = FixedWindows { patch, ops };
            track_level_impl(level, prev_points, displacements, windows, out)
        }
        match self {
            FixedKernel::Scalar => {
                run(level, prev_points, displacements, patch, fixed::Scalar, out)
            }
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sse2"
            ))]
            FixedKernel::Sse2 => run(level, prev_points, displacements, patch, fixed::Sse2, out),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            FixedKernel::Avx2(avx2) => run(level, prev_points, displacements, patch, avx2, out),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            FixedKernel::Simd128 => run(
                level,
                prev_points,
                displacements,
                patch,
                fixed::Simd128,
                out,
            ),
        }
    }
}

/// The kernel of one pass of the 8-bit tracker.
#[derive(Clone, Copy, Debug)]
enum ByteKernel {
    Reference(Kernel),
    Fixed(FixedKernel),
}

/// Per-window arithmetic of the Lucas-Kanade loop: the patch of the previous
/// frame, the residual sums against the next frame and the final error.
trait Windows<I: TrackImage> {
    /// Fills the patch of `win` and returns the spatial gradient matrix
    /// `(gxx, gxy, gyy)`, summed over the window, in (unit / pixel)^2.
    fn build(&mut self, lv: &Level<I>, win: &Window) -> (f32, f32, f32);

    /// The gradient-weighted residual `(bx, by)` between the patch and the
    /// next frame sampled at `win`.
    fn step(&self, lv: &Level<I>, win: &Window) -> (f32, f32);

    /// Mean absolute residual between the patch and the next frame at `win`.
    fn error(&self, lv: &Level<I>, win: &Window) -> f32;
}

/// The f32 reference windows.
struct FloatWindows<'a> {
    prev: &'a mut [f32],
    ix: &'a mut [f32],
    iy: &'a mut [f32],
}

impl<I: TrackImage> Windows<I> for FloatWindows<'_> {
    #[inline(always)]
    fn build(&mut self, lv: &Level<I>, win: &Window) -> (f32, f32, f32) {
        build_patch(lv, win, self.prev, self.ix, self.iy)
    }

    #[inline(always)]
    fn step(&self, lv: &Level<I>, win: &Window) -> (f32, f32) {
        lk_step(lv, win, self.prev, self.ix, self.iy)
    }

    #[inline(always)]
    fn error(&self, lv: &Level<I>, win: &Window) -> f32 {
        window_error(lv, win, self.prev)
    }
}

/// The fixed-point windows of the 8-bit tracker, with kernels `ops`.
struct FixedWindows<'a, K> {
    patch: fixed::Patch<'a>,
    ops: K,
}

impl<K: fixed::Ops> Windows<GrayImage> for FixedWindows<'_, K> {
    #[inline(always)]
    fn build(&mut self, lv: &Level<GrayImage>, win: &Window) -> (f32, f32, f32) {
        let source = fixed::Source {
            pixels: fixed::Plane {
                data: lv.prev,
                width: lv.width,
            },
            grad_x: fixed::Plane {
                data: lv.grad_x,
                width: lv.width,
            },
            grad_y: fixed::Plane {
                data: lv.grad_y,
                width: lv.width,
            },
        };
        let weights = fixed::Weights::of(win);
        let [xx, xy, yy] = self
            .ops
            .build(source, win.x0, win.y0, weights, &mut self.patch);
        let scale = |sum: i64| sum as f32 * fixed::SPATIAL_SCALE;
        (scale(xx), scale(xy), scale(yy))
    }

    #[inline(always)]
    fn step(&self, lv: &Level<GrayImage>, win: &Window) -> (f32, f32) {
        let next = fixed::Plane {
            data: lv.curr,
            width: lv.width,
        };
        let weights = fixed::Weights::of(win);
        let [bx, by] = self.ops.step(next, win.x0, win.y0, weights, &self.patch);
        (
            bx as f32 * fixed::RESIDUAL_SCALE,
            by as f32 * fixed::RESIDUAL_SCALE,
        )
    }

    #[inline(always)]
    fn error(&self, lv: &Level<GrayImage>, win: &Window) -> f32 {
        let next = fixed::Plane {
            data: lv.curr,
            width: lv.width,
        };
        let weights = fixed::Weights::of(win);
        let sum = self.ops.error(next, win.x0, win.y0, weights, &self.patch);
        sum as f32 * fixed::ERROR_SCALE / (lv.window_size * lv.window_size) as f32
    }
}

/// Refines every point's displacement at one pyramid level.
#[inline(always)]
fn track_level_impl<I: TrackImage, W: Windows<I>>(
    lv: &Level<I>,
    prev_points: &[(f32, f32)],
    displacements: &mut [(f32, f32)],
    mut windows: W,
    out: &mut [TrackResult],
) {
    let n_pixels = lv.window_size * lv.window_size;
    let window_limit = lv.window_size as f32;
    // Mean residual at the final position, infinite when the window left the
    // frame.
    let final_error = |windows: &W, cx: f32, cy: f32| {
        Window::new(cx, cy, lv.radius, lv.width, lv.height)
            .map_or(f32::INFINITY, |win| windows.error(lv, &win))
    };

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
        let (gxx, gxy, gyy) = windows.build(lv, &prev_win);

        // Reject low-texture windows up front (normalized by window area so
        // the threshold does not depend on `window_size`).
        let (inv_h00, inv_h01, inv_h11) =
            match I::spatial_inverse(gxx, gxy, gyy, n_pixels, lv.min_eigen_threshold) {
                Spatial::Invertible(inverse) => inverse,
                Spatial::LowTexture { measure_error } => {
                    out[idx].status = TrackStatus::LowTexture;
                    if measure_error && lv.is_finest {
                        out[idx].error = final_error(&windows, x + dx, y + dy);
                    }
                    continue;
                }
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

            let (bx, by) = windows.step(lv, &win);

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
                final_error(&windows, x + dx, y + dy)
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
fn build_patch<I: TrackImage>(
    lv: &Level<I>,
    win: &Window,
    prev_patch: &mut [f32],
    ix_patch: &mut [f32],
    iy_patch: &mut [f32],
) -> (f32, f32, f32) {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn lanes<P: Copy + Into<f32>, G: Copy + Into<f32>, const L: usize>(
        win: &Window,
        gwin: &Window,
        (pa, pb): (&[P], &[P]),
        (xa, xb): (&[G], &[G]),
        (ya, yb): (&[G], &[G]),
        p: &mut [f32],
        ox: &mut [f32],
        oy: &mut [f32],
        acc: &mut [[f32; L]; 3],
    ) {
        let s = sample_lanes::<P, L>(win, pa, pb);
        let sx = sample_lanes::<G, L>(gwin, xa, xb);
        let sy = sample_lanes::<G, L>(gwin, ya, yb);
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
    // Scharr outputs are 32x the intensity slope for both image types.
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
            lanes::<_, _, 8>(
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
            lanes::<_, _, 4>(
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
            lanes::<_, _, 1>(
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
fn lk_step<I: TrackImage>(
    lv: &Level<I>,
    win: &Window,
    prev_patch: &[f32],
    ix_patch: &[f32],
    iy_patch: &[f32],
) -> (f32, f32) {
    #[inline(always)]
    fn lanes<P: Copy + Into<f32>, const L: usize>(
        win: &Window,
        (a, b): (&[P], &[P]),
        p: &[f32],
        gx: &[f32],
        gy: &[f32],
        acc: &mut [[f32; L]; 2],
    ) {
        let s = sample_lanes::<P, L>(win, a, b);
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
            lanes::<_, 8>(
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
            lanes::<_, 4>(
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
            lanes::<_, 1>(
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
/// next image sampled at `win`.
#[inline(always)]
fn window_error<I: TrackImage>(lv: &Level<I>, win: &Window, prev_patch: &[f32]) -> f32 {
    #[inline(always)]
    fn lanes<P: Copy + Into<f32>, const L: usize>(
        win: &Window,
        (a, b): (&[P], &[P]),
        p: &[f32],
        acc: &mut [f32; L],
    ) {
        let s = sample_lanes::<P, L>(win, a, b);
        let p: &[f32; L] = p[..L].try_into().unwrap();
        for k in 0..L {
            acc[k] += (p[k] - s[k]).abs();
        }
    }

    let size = lv.window_size;
    let mut acc8 = [0f32; 8];
    let mut acc4 = [0f32; 4];
    let mut acc1 = [0f32; 1];

    for j in 0..size {
        let (a, b) = win.rows(lv.curr, lv.width, j, size);
        let p = &prev_patch[j * size..][..size];
        let mut i = 0;
        while i + 8 <= size {
            lanes::<_, 8>(win, (&a[i..], &b[i..]), &p[i..], &mut acc8);
            i += 8;
        }
        if i + 4 <= size {
            lanes::<_, 4>(win, (&a[i..], &b[i..]), &p[i..], &mut acc4);
            i += 4;
        }
        while i < size {
            lanes::<_, 1>(win, (&a[i..], &b[i..]), &p[i..], &mut acc1);
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
/// * `params` - as in [`calc_optical_flow_ex`]
/// * `fb_threshold` - maximum allowed round-trip distance in pixels; see
///   [`DEFAULT_FB_THRESHOLD`]
///
/// # Panics
/// As [`calc_optical_flow_ex`].
pub fn calc_optical_flow_fb(
    prev_pyramid: &[GrayImage],
    next_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    predicted: Option<&[(f32, f32)]>,
    params: TrackParams,
    fb_threshold: f32,
) -> Vec<TrackResult> {
    let mut scratch = Scratch::default();
    let mut forward = Vec::new();
    let kernel = GrayImage::select(TrackKernel::Auto, params.window_size);
    track_into(
        prev_pyramid,
        &mut Gradients::default(),
        next_pyramid,
        prev_points,
        predicted,
        params,
        None,
        kernel,
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
        &mut Gradients::default(),
        prev_pyramid,
        &forward_pos,
        Some(prev_points),
        params,
        None,
        kernel,
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

/// What the spatial gradient matrix of a window allows.
enum Spatial {
    /// The matrix inverse `(h00, h01, h11)`.
    Invertible((f32, f32, f32)),
    /// Too flat to track; `measure_error` tells whether the residual is
    /// still reported.
    LowTexture { measure_error: bool },
}

/// A pyramid level the Lucas-Kanade loop can track on: 8-bit images with
/// i16 Scharr gradients, or f32 images with f32 gradients.
trait TrackImage: Sized {
    type Pixel: Copy + Into<f32>;
    type Gradient: Copy + Into<f32> + Default;
    /// The window kernels a pass can run with.
    type Kernel: Copy;

    fn dimensions(&self) -> (u32, u32);

    fn pixels(&self) -> &[Self::Pixel];

    /// Scharr gradients, 32 times the slope per pixel, zero on the border.
    fn gradients_into(&self, grad_x: &mut [Self::Gradient], grad_y: &mut [Self::Gradient]);

    /// Classifies the spatial gradient matrix `[[gxx, gxy], [gxy, gyy]]`,
    /// summed over `n_pixels` window samples of slopes per pixel.
    fn spatial_inverse(gxx: f32, gxy: f32, gyy: f32, n_pixels: usize, threshold: f32) -> Spatial;

    /// The kernel for `choice` and windows of `window_size` on this CPU.
    fn select(choice: TrackKernel, window_size: usize) -> Self::Kernel;

    /// Refines every point's displacement at one level with `kernel`.
    fn track_level(
        kernel: Self::Kernel,
        level: &Level<Self>,
        prev_points: &[(f32, f32)],
        scratch: &mut Scratch,
        out: &mut [TrackResult],
    );
}

impl TrackImage for GrayImage {
    type Pixel = u8;
    type Gradient = i16;
    type Kernel = ByteKernel;

    fn dimensions(&self) -> (u32, u32) {
        ImageBuffer::dimensions(self)
    }

    fn select(choice: TrackKernel, window_size: usize) -> ByteKernel {
        let fixed = FixedKernel::detect();
        let fixed_point = window_size <= fixed::MAX_WINDOW
            && match choice {
                TrackKernel::Auto => !matches!(fixed, FixedKernel::Scalar),
                TrackKernel::FixedPoint => true,
                TrackKernel::Reference => false,
            };
        if fixed_point {
            ByteKernel::Fixed(fixed)
        } else {
            ByteKernel::Reference(Kernel::detect())
        }
    }

    fn track_level(
        kernel: ByteKernel,
        level: &Level<Self>,
        prev_points: &[(f32, f32)],
        scratch: &mut Scratch,
        out: &mut [TrackResult],
    ) {
        match kernel {
            ByteKernel::Reference(kernel) => kernel.track_level(level, prev_points, scratch, out),
            ByteKernel::Fixed(kernel) => kernel.track_level(level, prev_points, scratch, out),
        }
    }

    fn pixels(&self) -> &[u8] {
        self.as_raw()
    }

    fn gradients_into(&self, grad_x: &mut [i16], grad_y: &mut [i16]) {
        compute_gradients_into(self, grad_x, grad_y);
    }

    #[inline(always)]
    fn spatial_inverse(gxx: f32, gxy: f32, gyy: f32, n_pixels: usize, threshold: f32) -> Spatial {
        // Structured like OpenCV's calcOpticalFlowPyrLK: the smaller
        // eigenvalue per window sample, (trace - discriminant) / 2 in f32,
        // against the threshold, then a determinant test. It is kept so that
        // 8-bit results stay comparable with OpenCV and earlier releases.
        // As in OpenCV, the subtraction cancels for edge-like windows; see
        // DEFAULT_MIN_EIGEN_THRESHOLD for the resulting noise. Unlike OpenCV,
        // the discriminant is trace^2 - 4 det rather than (a - c)^2 + 4 b^2,
        // which costs precision only for nearly isotropic windows, below
        // 1e-3 relative, too little to move a decision. Sums and the
        // determinant bound are in (grey levels / pixel)^2, not OpenCV's
        // fixed-point scale. The f32 tracker uses a scale-free test instead.
        let min_eig = min_eigenvalue(gxx, gxy, gyy) / n_pixels as f32;
        if min_eig < threshold {
            return Spatial::LowTexture {
                measure_error: true,
            };
        }
        match invert_2x2(gxx, gxy, gyy, 1e-6) {
            Some(inverse) => Spatial::Invertible(inverse),
            None => Spatial::LowTexture {
                measure_error: false,
            },
        }
    }
}

impl TrackImage for FloatPyramidLevel {
    type Pixel = f32;
    type Gradient = f32;
    type Kernel = Kernel;

    fn dimensions(&self) -> (u32, u32) {
        ImageBuffer::dimensions(self)
    }

    fn select(_: TrackKernel, _: usize) -> Kernel {
        Kernel::detect()
    }

    fn track_level(
        kernel: Kernel,
        level: &Level<Self>,
        prev_points: &[(f32, f32)],
        scratch: &mut Scratch,
        out: &mut [TrackResult],
    ) {
        kernel.track_level(level, prev_points, scratch, out)
    }

    fn pixels(&self) -> &[f32] {
        self.as_raw()
    }

    fn gradients_into(&self, grad_x: &mut [f32], grad_y: &mut [f32]) {
        let (width, height) = ImageBuffer::dimensions(self);
        scharr_f32(
            self.as_raw(),
            width as usize,
            height as usize,
            grad_x,
            grad_y,
        );
    }

    #[inline(always)]
    fn spatial_inverse(gxx: f32, gxy: f32, gyy: f32, n_pixels: usize, threshold: f32) -> Spatial {
        // Physical units can make every entry tiny, so no test here uses an
        // absolute epsilon. In f64, det / lambda_max has no cancellation, and
        // an eigenvalue ratio below f32 resolution means a singular matrix
        // at any scale.
        let (a, b, c) = (gxx as f64, gyy as f64, gxy as f64);
        let determinant = a * b - c * c;
        let largest = (a + b + ((a - b).powi(2) + 4.0 * c * c).sqrt()) * 0.5;
        let smallest = if determinant > 0.0 {
            determinant / largest
        } else {
            0.0
        };
        let resolvable = smallest > largest * f32::EPSILON as f64;
        if !resolvable || smallest / (n_pixels as f64) < threshold as f64 {
            return Spatial::LowTexture {
                measure_error: true,
            };
        }
        let inverse = 1.0 / determinant;
        Spatial::Invertible((
            (b * inverse) as f32,
            (-c * inverse) as f32,
            (a * inverse) as f32,
        ))
    }
}

/// Upper bounds for the work a tracker context will see, used to allocate
/// every buffer up front with `with_capacity` or `reserve` of
/// [`TrackerContext`] and [`FloatTrackerContext`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackerCapacity {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pyramid levels passed to `prepare` / `advance`.
    pub levels: usize,
    /// Largest `window_size` passed to `track` / `track_fb`.
    pub window_size: usize,
    /// Most points per `track` / `track_fb` call.
    pub points: usize,
}

/// Pyramids, their gradients, scratch and result buffers of a tracker
/// context, generic over the pyramid level type.
#[derive(Default)]
struct Tracker<I: TrackImage> {
    prev_pyramid: Vec<I>,
    next_pyramid: Vec<I>,
    /// Gradients of `prev_pyramid`, which the forward pass tracks from.
    prev_gradients: Gradients<I::Gradient>,
    /// Gradients of `next_pyramid`, which the backward pass tracks from and
    /// which become `prev_gradients` when `advance` reuses the pyramid.
    next_gradients: Gradients<I::Gradient>,
    /// Level count of `next_pyramid` while it holds the pyramid of the last
    /// next frame, which `advance` can then reuse. `None` before the first
    /// frame and after `reserve` reshaped the base level.
    next_levels: Option<usize>,
    scratch: Scratch,
    results: Vec<TrackResult>,
    forward_pos: Vec<(f32, f32)>,
    backward: Vec<TrackResult>,
}

impl<P> Tracker<ImageBuffer<P, Vec<P::Subpixel>>>
where
    P: Pixel,
    P::Subpixel: Default,
    ImageBuffer<P, Vec<P::Subpixel>>: TrackImage,
{
    /// Allocates both pyramids and every buffer for `capacity`.
    fn reserve(&mut self, capacity: &TrackerCapacity) {
        let TrackerCapacity {
            width,
            height,
            levels,
            points,
            ..
        } = *capacity;
        reserve_pyramid(&mut self.prev_pyramid, width, height, levels);
        let base = self.next_pyramid.first().map(|level| level.dimensions());
        reserve_pyramid(&mut self.next_pyramid, width, height, levels);
        if self.next_pyramid.first().map(|level| level.dimensions()) != base {
            // The base level was reshaped and no longer holds the last frame.
            self.next_levels = None;
        }
        // Levels may have been reshaped, so their gradients are recomputed
        // when next needed.
        for gradients in [&mut self.prev_gradients, &mut self.next_gradients] {
            gradients.reserve(width, height, levels);
            gradients.invalidate();
        }
        self.scratch.reserve(capacity);
        reserve_total(&mut self.results, points);
        reserve_total(&mut self.forward_pos, points);
        reserve_total(&mut self.backward, points);
    }
}

impl<I: TrackImage> Tracker<I> {
    /// Records that both pyramids were just built from a frame pair.
    fn built_both(&mut self) {
        self.prev_gradients.invalidate();
        self.next_gradients.invalidate();
        self.built_next();
    }

    /// Records that `next_pyramid` was just built from a frame.
    fn built_next(&mut self) {
        self.next_gradients.invalidate();
        self.next_levels = Some(self.next_pyramid.len());
    }

    /// Shared body of `advance`. `build` builds a pyramid of the new frame of
    /// size `dimensions`, `rebuild` one of the given base level, both with the
    /// caller's `levels`.
    fn advance(
        &mut self,
        dimensions: (u32, u32),
        levels: usize,
        build: impl Fn(&mut Vec<I>),
        rebuild: impl FnOnce(&I, &mut Vec<I>),
    ) {
        // The base level of the next pyramid is the last frame itself.
        let last_frame = self
            .next_levels
            .filter(|_| self.next_pyramid.first().map(I::dimensions) == Some(dimensions));
        match last_frame {
            Some(built)
                if built == self.next_pyramid.len()
                    && built == pyramid_level_count(dimensions.0, dimensions.1, levels) =>
            {
                // The last frame's gradients move along with its pyramid.
                std::mem::swap(&mut self.prev_pyramid, &mut self.next_pyramid);
                std::mem::swap(&mut self.prev_gradients, &mut self.next_gradients);
            }
            Some(_) => {
                rebuild(&self.next_pyramid[0], &mut self.prev_pyramid);
                self.prev_gradients.invalidate();
            }
            None => {
                build(&mut self.prev_pyramid);
                self.prev_gradients.invalidate();
            }
        }
        build(&mut self.next_pyramid);
        self.built_next();
    }

    fn track(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
        options: TrackOptions,
    ) -> &[TrackResult] {
        track_into(
            &self.prev_pyramid,
            &mut self.prev_gradients,
            &self.next_pyramid,
            prev_points,
            predicted,
            params,
            options.start_level,
            I::select(options.kernel, params.window_size),
            &mut self.scratch,
            &mut self.results,
        );
        &self.results
    }

    fn track_fb(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
        fb_threshold: f32,
        options: TrackOptions,
    ) -> &[TrackResult] {
        let backward_start = options.fb_levels.map(|levels| {
            assert!(levels >= 1, "fb_levels must be at least 1");
            levels - 1
        });
        let kernel = I::select(options.kernel, params.window_size);
        track_into(
            &self.prev_pyramid,
            &mut self.prev_gradients,
            &self.next_pyramid,
            prev_points,
            predicted,
            params,
            options.start_level,
            kernel,
            &mut self.scratch,
            &mut self.results,
        );

        self.forward_pos.clear();
        self.forward_pos.extend(self.results.iter().map(|r| r.pos));

        // Seed the backward pass at the original points (see calc_optical_flow_fb).
        track_into(
            &self.next_pyramid,
            &mut self.next_gradients,
            &self.prev_pyramid,
            &self.forward_pos,
            Some(prev_points),
            params,
            backward_start,
            kernel,
            &mut self.scratch,
            &mut self.backward,
        );

        mark_fb_inconsistent(&mut self.results, &self.backward, prev_points, fb_threshold);
        &self.results
    }
}

/// Reusable owner of every buffer the tracking hot path touches: both frame
/// pyramids, the Lucas-Kanade scratch, the result vector and the
/// forward-backward intermediates.
///
/// Create one per tracking thread. Each frame, call [`advance`](Self::advance)
/// with the new frame, or [`prepare`](Self::prepare) with a frame pair, then
/// [`track`](Self::track) / [`track_fb`](Self::track_fb). `advance` reuses
/// the pyramid of the last frame, so it builds one pyramid per frame instead
/// of two. All buffers are resized in place and reused. To keep allocation
/// out of the per-frame path, create the context with
/// [`with_capacity`](Self::with_capacity): frames of the reserved size and
/// level count, with at most the reserved window size and point count, then
/// perform **no heap allocation**. Without a reservation the first frame
/// allocates, and so does any later call with more points than before. The
/// free functions ([`calc_optical_flow_ex`], [`calc_optical_flow_fb`]) are
/// thin convenience wrappers that allocate their own scratch.
#[derive(Default)]
pub struct TrackerContext {
    tracker: Tracker<GrayImage>,
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
        self.tracker.reserve(&capacity);
    }

    /// Builds the previous- and next-frame pyramids into the context's reusable
    /// buffers. Zero-alloc in steady state (same image size and `levels`).
    ///
    /// # Panics
    /// Panics if the two frames differ in size.
    pub fn prepare(&mut self, prev: &GrayImage, next: &GrayImage, levels: usize) {
        assert_eq!(
            prev.dimensions(),
            next.dimensions(),
            "frames must have the same size"
        );
        build_pyramid_into(prev, levels, &mut self.tracker.prev_pyramid);
        build_pyramid_into(next, levels, &mut self.tracker.next_pyramid);
        self.tracker.built_both();
    }

    /// Moves to the next frame pair: the pyramid of the last next frame
    /// becomes the previous-frame pyramid, by swapping buffers, and the
    /// next-frame pyramid is built from `frame`.
    ///
    /// The pyramids, and therefore all tracking results, equal those of
    /// `prepare(last_next, frame, levels)` bit for bit, where `last_next` is
    /// the last frame passed to `advance` or as `next` to `prepare`. Only one
    /// pyramid is built, and the caller does not need to keep the last frame.
    /// This is the preferred per-frame call. In steady state (same frame size
    /// and `levels`) it performs no heap allocation.
    ///
    /// When the last frame cannot be reused as is, the call still leaves both
    /// pyramids ready for tracking:
    /// - With a different `levels` than before, the previous-frame pyramid is
    ///   rebuilt from the last frame, which still equals
    ///   `prepare(last_next, frame, levels)`.
    /// - Without a last frame (on the first call, or after
    ///   [`reserve`](Self::reserve) changed the frame size) or when `frame`
    ///   differs in size from it, both pyramids are built from `frame`, as
    ///   `prepare(frame, frame, levels)` does. Tracking on that pair measures
    ///   no motion.
    ///
    /// ```
    /// # use image::GrayImage;
    /// # use optical_flow_lk::{DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, TrackParams, TrackerContext};
    /// # let frames = vec![GrayImage::new(64, 48); 3];
    /// # let mut points = vec![(32.0, 24.0)];
    /// # let params = TrackParams {
    /// #     window_size: 21,
    /// #     max_iterations: 30,
    /// #     min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
    /// # };
    /// let mut ctx = TrackerContext::new();
    /// ctx.advance(&frames[0], 4);
    /// for frame in &frames[1..] {
    ///     ctx.advance(frame, 4);
    ///     let results = ctx.track_fb(&points, None, params, DEFAULT_FB_THRESHOLD);
    ///     points = results.iter().map(|result| result.pos).collect();
    /// }
    /// ```
    pub fn advance(&mut self, frame: &GrayImage, levels: usize) {
        self.tracker.advance(
            frame.dimensions(),
            levels,
            |pyramid| build_pyramid_into(frame, levels, pyramid),
            |base, pyramid| build_pyramid_into(base, levels, pyramid),
        );
    }

    /// The previous-frame pyramid built by the last [`prepare`](Self::prepare)
    /// or [`advance`](Self::advance).
    pub fn prev_pyramid(&self) -> &[GrayImage] {
        &self.tracker.prev_pyramid
    }

    /// The next-frame pyramid built by the last [`prepare`](Self::prepare) or
    /// [`advance`](Self::advance).
    pub fn next_pyramid(&self) -> &[GrayImage] {
        &self.tracker.next_pyramid
    }

    /// Tracks `prev_points` using the prepared pyramids, returning the results
    /// held inside the context. See [`calc_optical_flow_ex`] for the argument
    /// semantics. Allocation-free in steady state.
    ///
    /// The Scharr gradients of each pyramid level are computed once per frame
    /// and kept with the pyramid, so `track` after `track_fb` on the same
    /// frames, or the next frame's forward pass after `advance`, reuse them.
    pub fn track(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
    ) -> &[TrackResult] {
        self.track_with(prev_points, predicted, params, TrackOptions::new())
    }

    /// [`track`](Self::track) with [`TrackOptions`]: the level the pass starts
    /// at and the kernel arithmetic.
    pub fn track_with(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
        options: TrackOptions,
    ) -> &[TrackResult] {
        self.tracker.track(prev_points, predicted, params, options)
    }

    /// Forward-backward consistent tracking using the prepared pyramids. See
    /// [`calc_optical_flow_fb`] for semantics. Reuses the context's scratch and
    /// intermediate point buffers, so it is allocation-free in steady state.
    pub fn track_fb(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
        fb_threshold: f32,
    ) -> &[TrackResult] {
        self.track_fb_with(
            prev_points,
            predicted,
            params,
            fb_threshold,
            TrackOptions::new(),
        )
    }

    /// [`track_fb`](Self::track_fb) with [`TrackOptions`]: the level the
    /// forward pass starts at, the levels of the backward pass and the kernel
    /// arithmetic.
    ///
    /// ```
    /// # use image::GrayImage;
    /// # use optical_flow_lk::{DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, TrackOptions, TrackParams, TrackerContext};
    /// # let (frame, points) = (GrayImage::new(64, 48), vec![(32.0, 24.0)]);
    /// # let params = TrackParams {
    /// #     window_size: 13,
    /// #     max_iterations: 30,
    /// #     min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
    /// # };
    /// let mut ctx = TrackerContext::new();
    /// ctx.advance(&frame, 4);
    /// // Round trip over the finest level only.
    /// let options = TrackOptions::new().with_fb_levels(1);
    /// let results = ctx.track_fb_with(&points, None, params, DEFAULT_FB_THRESHOLD, options);
    /// ```
    ///
    /// # Panics
    /// As [`track`](Self::track), and if `options.fb_levels` is `Some(0)`.
    pub fn track_fb_with(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
        fb_threshold: f32,
        options: TrackOptions,
    ) -> &[TrackResult] {
        self.tracker
            .track_fb(prev_points, predicted, params, fb_threshold, options)
    }
}

/// Pyramidal Lucas-Kanade tracker for f32 images in physical units, e.g.
/// thermal frames in °C. The counterpart of [`TrackerContext`], owning both
/// frame pyramids and every buffer the tracking hot path touches.
///
/// The algorithm and its parameters match the 8-bit tracker: pyramids of
/// 2x2 means, Scharr gradients, bilinear window sampling, the same
/// convergence rules, prediction and forward-backward check, and the same
/// [`TrackResult`] and [`TrackStatus`]. Pixels, pyramids and gradients stay
/// in f32 throughout; nothing is quantized. The pyramid averages exactly
/// instead of truncating like the 8-bit pyramid, so on the same frames the
/// two trackers agree closely but not bit for bit.
///
/// Units follow the image:
/// - [`TrackParams::min_eigen_threshold`] is in (input unit / pixel)^2, the
///   unit of the f32 detector's scores, and has no default. A window whose mean minimum
///   gradient eigenvalue is below it is [`TrackStatus::LowTexture`]. Windows
///   whose eigenvalue ratio is below f32 resolution are low texture as well,
///   at any contrast: the singularity test is relative, so small physical
///   gradients still track.
/// - [`TrackResult::error`] is the mean absolute residual in the input unit.
/// - Positions, `fb_threshold` and [`CONVERGENCE_EPSILON`] are in pixels.
///
/// Per frame, call [`advance`](Self::advance) with the new frame, or
/// [`prepare`](Self::prepare) with a frame pair, as for [`TrackerContext`].
/// Create the context with [`with_capacity`](Self::with_capacity) to keep
/// allocation out of the per-frame path.
#[derive(Default)]
pub struct FloatTrackerContext {
    tracker: Tracker<FloatPyramidLevel>,
}

impl FloatTrackerContext {
    /// Creates an empty context. Buffers grow to fit on the first call and
    /// are reused thereafter.
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
        self.tracker.reserve(&capacity);
    }

    /// Builds the previous- and next-frame pyramids with up to `levels`
    /// levels; halving stops below 2 pixels.
    ///
    /// # Panics
    /// Panics if the two frames differ in size.
    pub fn prepare(&mut self, prev: FloatImageView<'_>, next: FloatImageView<'_>, levels: usize) {
        assert_eq!(
            prev.dimensions(),
            next.dimensions(),
            "frames must have the same size"
        );
        build_float_pyramid_into(prev, levels, &mut self.tracker.prev_pyramid);
        build_float_pyramid_into(next, levels, &mut self.tracker.next_pyramid);
        self.tracker.built_both();
    }

    /// Moves to the next frame pair: the pyramid of the last next frame
    /// becomes the previous-frame pyramid, by swapping buffers, and the
    /// next-frame pyramid is built from `frame`. The results equal those of
    /// `prepare(last_next, frame, levels)` bit for bit with one pyramid build
    /// instead of two, and the steady state performs no heap allocation.
    /// Without a reusable last frame, the pyramids are built as described
    /// for [`TrackerContext::advance`].
    pub fn advance(&mut self, frame: FloatImageView<'_>, levels: usize) {
        self.tracker.advance(
            frame.dimensions(),
            levels,
            |pyramid| build_float_pyramid_into(frame, levels, pyramid),
            |base, pyramid| {
                let (width, height) = base.dimensions();
                let base = FloatImageView::new(base.as_raw(), width, height)
                    .expect("the base level is a copy of a validated frame");
                build_float_pyramid_into(base, levels, pyramid);
            },
        );
    }

    /// The previous-frame pyramid built by the last [`prepare`](Self::prepare)
    /// or [`advance`](Self::advance).
    pub fn prev_pyramid(&self) -> &[FloatPyramidLevel] {
        &self.tracker.prev_pyramid
    }

    /// The next-frame pyramid built by the last [`prepare`](Self::prepare) or
    /// [`advance`](Self::advance).
    pub fn next_pyramid(&self) -> &[FloatPyramidLevel] {
        &self.tracker.next_pyramid
    }

    /// Tracks `prev_points` from the previous into the next frame, as
    /// [`calc_optical_flow_ex`] does. `params.min_eigen_threshold` is in
    /// (input unit / pixel)^2. Returns one result per point, in order, from
    /// the context's own buffer.
    ///
    /// # Panics
    /// Panics if `prepare` was not called, `params.window_size` is even, or
    /// `predicted` has a different length than `prev_points`.
    pub fn track(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
    ) -> &[TrackResult] {
        self.track_with(prev_points, predicted, params, TrackOptions::new())
    }

    /// [`track`](Self::track) with [`TrackOptions`]. The f32 tracker always
    /// uses its f32 kernels, so only `start_level` applies.
    pub fn track_with(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
        options: TrackOptions,
    ) -> &[TrackResult] {
        self.tracker.track(prev_points, predicted, params, options)
    }

    /// Forward-backward consistent tracking, as [`calc_optical_flow_fb`]
    /// does; `fb_threshold` is in pixels.
    ///
    /// # Panics
    /// As [`track`](Self::track).
    pub fn track_fb(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
        fb_threshold: f32,
    ) -> &[TrackResult] {
        self.track_fb_with(
            prev_points,
            predicted,
            params,
            fb_threshold,
            TrackOptions::new(),
        )
    }

    /// [`track_fb`](Self::track_fb) with [`TrackOptions`]: the level the
    /// forward pass starts at and the levels of the backward pass.
    ///
    /// # Panics
    /// As [`track`](Self::track), and if `options.fb_levels` is `Some(0)`.
    pub fn track_fb_with(
        &mut self,
        prev_points: &[(f32, f32)],
        predicted: Option<&[(f32, f32)]>,
        params: TrackParams,
        fb_threshold: f32,
        options: TrackOptions,
    ) -> &[TrackResult] {
        self.tracker
            .track_fb(prev_points, predicted, params, fb_threshold, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::build_pyramid;
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
            let lv = Level::<GrayImage> {
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
            assert!(window_error(&lv, &win, &patch) < 1e-4);
            let (bx, by) = lk_step(&lv, &win, &patch, &ix, &iy);
            assert!(
                bx.abs() < 1e-2 && by.abs() < 1e-2,
                "size {size}: ({bx}, {by})"
            );
        }
    }

    /// Every fixed-point kernel this CPU runs.
    fn fixed_kernels() -> Vec<FixedKernel> {
        #[allow(unused_mut)]
        let mut kernels = vec![FixedKernel::Scalar, FixedKernel::BASELINE];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(avx2) = fixed::Avx2::detect() {
            kernels.push(FixedKernel::Avx2(avx2));
        }
        kernels
    }

    /// Smoothed hash noise, `w` x `h`, shifted by `(sx, sy)` pixels.
    fn texture(w: u32, h: u32, (sx, sy): (f32, f32)) -> GrayImage {
        let noise = |x: i32, y: i32| {
            let mut s = (y.rem_euclid(1024) as u32 * 1024 + x.rem_euclid(1024) as u32)
                .wrapping_mul(2654435761);
            s ^= s >> 15;
            s = s.wrapping_mul(0x85ebca6b);
            (s >> 24) as f32
        };
        let smooth = |x: f32, y: f32| {
            let (x0, y0) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0);
            let mut total = 0.0;
            for dy in -2..=3 {
                for dx in -2..=3 {
                    let wx = if dx == -2 {
                        1.0 - fx
                    } else if dx == 3 {
                        fx
                    } else {
                        1.0
                    };
                    let wy = if dy == -2 {
                        1.0 - fy
                    } else if dy == 3 {
                        fy
                    } else {
                        1.0
                    };
                    total += wx * wy * noise(x0 as i32 + dx, y0 as i32 + dy);
                }
            }
            total / 25.0
        };
        GrayImage::from_fn(w, h, |x, y| {
            Luma([smooth(x as f32 - sx, y as f32 - sy) as u8])
        })
    }

    #[test]
    fn fixed_point_kernels_track_identically() {
        for (w, h, shift) in [(97u32, 71u32, (1.7f32, -0.6f32)), (160, 120, (4.25, 3.5))] {
            let prev = build_pyramid(&texture(w, h, (0.0, 0.0)), 3);
            let next = build_pyramid(&texture(w, h, shift), 3);
            // A lattice reaching into the borders, where windows fall back
            // to the scalar kernel and leave the frame.
            let points: Vec<(f32, f32)> = (0..90)
                .map(|i| {
                    (
                        1.5 + (i % 10) as f32 * w as f32 / 9.6,
                        2.0 + (i / 10) as f32 * h as f32 / 8.7,
                    )
                })
                .collect();
            let predicted: Vec<(f32, f32)> = points
                .iter()
                .map(|&(x, y)| (x + shift.0, y + shift.1))
                .collect();
            for window_size in [1, 7, 13, 21, 31] {
                let params = TrackParams {
                    window_size,
                    max_iterations: 20,
                    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
                };
                for guess in [None, Some(predicted.as_slice())] {
                    let run = |kernel| {
                        let mut out = Vec::new();
                        track_into(
                            &prev,
                            &mut Gradients::default(),
                            &next,
                            &points,
                            guess,
                            params,
                            None,
                            ByteKernel::Fixed(kernel),
                            &mut Scratch::default(),
                            &mut out,
                        );
                        out.iter()
                            .map(|r| {
                                (
                                    r.pos.0.to_bits(),
                                    r.pos.1.to_bits(),
                                    r.status,
                                    r.error.to_bits(),
                                )
                            })
                            .collect::<Vec<_>>()
                    };
                    let kernels = fixed_kernels();
                    let scalar = run(kernels[0]);
                    let tracked = scalar
                        .iter()
                        .filter(|r| r.2 == TrackStatus::Tracked)
                        .count();
                    assert!(window_size == 1 || tracked > 20, "{tracked} tracked");
                    for &kernel in &kernels[1..] {
                        assert_eq!(
                            run(kernel),
                            scalar,
                            "{kernel:?}, {w}x{h}, window {window_size}"
                        );
                    }
                }
            }
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
