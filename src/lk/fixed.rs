//! Fixed-point window kernels of the 8-bit tracker.
//!
//! The scheme follows OpenCV's `calcOpticalFlowPyrLK`: patches and gradients
//! are 16-bit integers, the bilinear weights are 14-bit integers, and the
//! sums are formed from 16-bit products accumulated in 32-bit lanes:
//! pairwise multiply-adds (`_mm_madd_epi16`, `_mm256_madd_epi16`) on x86,
//! widening multiplies on wasm32 `simd128`. Every kernel computes exactly the
//! integers defined by [`Scalar`], so the results are identical on every
//! target and instruction set.
//!
//! # Numerics
//!
//! For a window whose samples share the fractional offset `(fx, fy)`:
//!
//! - Weights: `w00 = round(2^14 (1 - fx) (1 - fy))`, `w10 = round(2^14 fx
//!   (1 - fy))`, `w01 = round(2^14 (1 - fx) fy)` and `w11 = 2^14 - w00 - w10 -
//!   w01`, so the weights sum to exactly `2^14`. Rounding can leave `w11 = -1`
//!   when `fx * fy` is tiny; the bounds below include that case.
//! - Patch intensities: `I = (a0 w00 + a1 w10 + b0 w01 + b1 w11 + 2^8) >> 9`,
//!   32 times the grey level (5 fractional bits), in `0..=8160`.
//! - Patch gradients: the same sum over the Scharr planes, `+ 2^12 >> 13`,
//!   twice the Scharr scale: 64 times the slope per pixel (OpenCV keeps the
//!   Scharr scale; the extra bit halves the quantization of weak gradients,
//!   which the coarse pyramid levels of real frames have). 8-bit Scharr
//!   responses are at most `16 * 255`, so `|g| <= 8160`.
//! - Residuals `d = I - J`, with `J` the next frame interpolated like `I`:
//!   `|d| <= 8160`.
//! - Sums `Σ gx^2, Σ gx gy, Σ gy^2`, `Σ d gx, Σ d gy` and `Σ |d|` are exact
//!   integers. Kernels accumulate in i32 lanes and move the lanes into i64
//!   totals before they can overflow (see [`PAIRS`]); integer addition is
//!   associative, so the lane layout and flush schedule do not change the
//!   result.
//!
//! The caller converts the sums to the f32 units of the reference kernel:
//! `Σ g^2 / 4096` is in (grey levels / pixel)^2, `Σ d g / 2048` in grey levels
//! per pixel, `Σ |d| / 32` in grey levels.
//!
//! The bounds take `w11 = -1` into account: the weights' absolute values then
//! sum to `2^14 + 2`, and the rounded sums still stay within `0..=8160` and
//! `-8160..=8160`.
//!
//! # Overflow
//!
//! A pairwise multiply-add of two products adds at most `2 * 8160^2 =
//! 133_171_200` to a lane, for the spatial and the residual sums alike, so a
//! lane holds 16 of them. A window row of `size` pixels feeds each lane at
//! most `stride / 8` of them, where the patch stride is `size` rounded up to
//! 8; kernels flush the lanes after `16 / (stride / 8)` rows. That leaves at
//! least one row per flush up to a stride of 128, hence [`MAX_WINDOW`]: a
//! 13 px window (stride 16, two pairs per lane and row) flushes once within
//! the window, a 21 px window (stride 24, three) every 5 rows. `Σ |d|` is at
//! most `127^2 * 8160 < 2^31` and never needs a flush. On 8-bit input the
//! bounds hold for any image content: the adversarial cases in the tests
//! saturate the spatial bound.

use super::Window;

/// Fractional bits of the bilinear weights.
const WEIGHT_BITS: u32 = 14;
const WEIGHT_ONE: i32 = 1 << WEIGHT_BITS;
/// Patch intensities keep 5 fractional bits: 32 x the grey level.
const PIXEL_SHIFT: u32 = WEIGHT_BITS - 5;
const PIXEL_ROUND: i32 = 1 << (PIXEL_SHIFT - 1);
/// Patch gradients keep one bit more than the Scharr planes: 64 x the slope.
const GRADIENT_SHIFT: u32 = WEIGHT_BITS - 1;
const GRADIENT_ROUND: i32 = 1 << (GRADIENT_SHIFT - 1);

/// Largest interpolated |gradient| and intensity (see the module docs).
const GRADIENT_MAX: i64 = 2 * 16 * 255;
const PIXEL_MAX: i64 = 32 * 255;
/// Pairwise products one i32 lane holds without overflow, of the spatial
/// (`g g`) and the residual (`d g`) terms alike.
const PAIRS: usize = {
    let other = if PIXEL_MAX > GRADIENT_MAX {
        PIXEL_MAX
    } else {
        GRADIENT_MAX
    };
    (i32::MAX as i64 / (2 * GRADIENT_MAX * other)) as usize
};
/// Largest window the fixed-point kernels take: every row must fit between
/// two flushes, so the stride is at most `8 * PAIRS`.
pub(super) const MAX_WINDOW: usize = 8 * PAIRS - 1;

const _: () = assert!(PAIRS == 16 && MAX_WINDOW == 127);

/// Scale from the integer spatial sums to (grey levels / pixel)^2.
pub(super) const SPATIAL_SCALE: f32 = 1.0 / 4096.0;
/// Scale from the integer residual sums to grey levels per pixel.
pub(super) const RESIDUAL_SCALE: f32 = 1.0 / 2048.0;
/// Scale from `Σ |d|` to grey levels.
pub(super) const ERROR_SCALE: f32 = 1.0 / 32.0;

/// Row stride of a patch: the window size rounded up to whole 8-lane chunks.
pub(super) fn stride(size: usize) -> usize {
    size.next_multiple_of(8)
}

/// The 14-bit bilinear weights of a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Weights {
    w00: i32,
    w10: i32,
    w01: i32,
    w11: i32,
}

impl Weights {
    pub(super) fn of(win: &Window) -> Self {
        let round = |weight: f32| (weight * WEIGHT_ONE as f32 + 0.5) as i32;
        let (w00, w10, w01) = (round(win.w00), round(win.w10), round(win.w01));
        Weights {
            w00,
            w10,
            w01,
            w11: WEIGHT_ONE - w00 - w10 - w01,
        }
    }

    /// The bilinear sum of element `i` of two source rows.
    #[inline(always)]
    fn sum<T: Copy + Into<i32>>(&self, a: &[T], b: &[T], i: usize) -> i32 {
        a[i].into() * self.w00
            + a[i + 1].into() * self.w10
            + b[i].into() * self.w01
            + b[i + 1].into() * self.w11
    }
}

/// One plane of a pyramid level, row-major with `width` elements per row.
#[derive(Clone, Copy)]
pub(super) struct Plane<'a, T> {
    pub(super) data: &'a [T],
    pub(super) width: usize,
}

impl<'a, T> Plane<'a, T> {
    /// The two rows feeding window row `j` of a window at `(x0, y0)`: `span`
    /// elements each.
    #[inline(always)]
    fn rows(&self, x0: usize, y0: usize, j: usize, span: usize) -> (&'a [T], &'a [T]) {
        let start = (y0 + j) * self.width + x0;
        let below = start + self.width;
        (
            &self.data[start..start + span],
            &self.data[below..below + span],
        )
    }

    /// Whether every source row of a `size` window at `(x0, y0)` has `span`
    /// readable elements, which the vector kernels read past the window's own
    /// `size + 1`.
    #[inline(always)]
    fn holds(&self, x0: usize, y0: usize, size: usize, span: usize) -> bool {
        (y0 + size) * self.width + x0 + span <= self.data.len()
    }
}

/// The previous frame's level: pixels and Scharr gradients.
#[derive(Clone, Copy)]
pub(super) struct Source<'a> {
    pub(super) pixels: Plane<'a, u8>,
    pub(super) grad_x: Plane<'a, i16>,
    pub(super) grad_y: Plane<'a, i16>,
}

/// The fixed-point patch of one window of the previous frame: `size` rows of
/// `stride` values each, zero past `size`.
pub(super) struct Patch<'a> {
    pub(super) size: usize,
    pub(super) stride: usize,
    pub(super) pixels: &'a mut [i16],
    pub(super) grad_x: &'a mut [i16],
    pub(super) grad_y: &'a mut [i16],
}

/// The window kernels of one instruction set.
pub(super) trait Ops: Copy {
    /// Fills `patch` from the window at `(x0, y0)` with weights `w` and
    /// returns `[Σ gx^2, Σ gx gy, Σ gy^2]`.
    fn build(
        self,
        src: Source<'_>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &mut Patch<'_>,
    ) -> [i64; 3];

    /// `[Σ d gx, Σ d gy]` against the next frame sampled at `(x0, y0)`.
    fn step(
        self,
        next: Plane<'_, u8>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &Patch<'_>,
    ) -> [i64; 2];

    /// `Σ |d|` against the next frame sampled at `(x0, y0)`.
    fn error(self, next: Plane<'_, u8>, x0: usize, y0: usize, w: Weights, patch: &Patch<'_>)
    -> i64;
}

/// The definition of the fixed-point kernels, one pixel at a time. Vector
/// kernels fall back to it for windows whose over-read would leave the frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct Scalar;

impl Ops for Scalar {
    fn build(
        self,
        src: Source<'_>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &mut Patch<'_>,
    ) -> [i64; 3] {
        let (size, stride) = (patch.size, patch.stride);
        let mut sums = [0i64; 3];
        for j in 0..size {
            let (pa, pb) = src.pixels.rows(x0, y0, j, size + 1);
            let (xa, xb) = src.grad_x.rows(x0, y0, j, size + 1);
            let (ya, yb) = src.grad_y.rows(x0, y0, j, size + 1);
            let row = j * stride..(j + 1) * stride;
            let pixels = &mut patch.pixels[row.clone()];
            let grad_x = &mut patch.grad_x[row.clone()];
            let grad_y = &mut patch.grad_y[row];
            for i in 0..size {
                let p = (w.sum(pa, pb, i) + PIXEL_ROUND) >> PIXEL_SHIFT;
                let gx = (w.sum(xa, xb, i) + GRADIENT_ROUND) >> GRADIENT_SHIFT;
                let gy = (w.sum(ya, yb, i) + GRADIENT_ROUND) >> GRADIENT_SHIFT;
                pixels[i] = p as i16;
                grad_x[i] = gx as i16;
                grad_y[i] = gy as i16;
                sums[0] += (gx * gx) as i64;
                sums[1] += (gx * gy) as i64;
                sums[2] += (gy * gy) as i64;
            }
            for plane in [&mut *pixels, &mut *grad_x, &mut *grad_y] {
                plane[size..].fill(0);
            }
        }
        sums
    }

    fn step(
        self,
        next: Plane<'_, u8>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &Patch<'_>,
    ) -> [i64; 2] {
        let (size, stride) = (patch.size, patch.stride);
        let mut sums = [0i64; 2];
        for j in 0..size {
            let (a, b) = next.rows(x0, y0, j, size + 1);
            let row = j * stride..j * stride + size;
            let pixels = &patch.pixels[row.clone()];
            let grad_x = &patch.grad_x[row.clone()];
            let grad_y = &patch.grad_y[row];
            for i in 0..size {
                let d = pixels[i] as i32 - ((w.sum(a, b, i) + PIXEL_ROUND) >> PIXEL_SHIFT);
                sums[0] += (d * grad_x[i] as i32) as i64;
                sums[1] += (d * grad_y[i] as i32) as i64;
            }
        }
        sums
    }

    fn error(
        self,
        next: Plane<'_, u8>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &Patch<'_>,
    ) -> i64 {
        let (size, stride) = (patch.size, patch.stride);
        let mut sum = 0i64;
        for j in 0..size {
            let (a, b) = next.rows(x0, y0, j, size + 1);
            let pixels = &patch.pixels[j * stride..j * stride + size];
            for (i, &p) in pixels.iter().enumerate() {
                let d = p as i32 - ((w.sum(a, b, i) + PIXEL_ROUND) >> PIXEL_SHIFT);
                sum += d.unsigned_abs() as i64;
            }
        }
        sum
    }
}

/// Rows of i32 lanes accumulated between flushes into i64 totals: [`PAIRS`]
/// pairwise products per lane, of which a row adds one per 8-lane chunk.
#[inline(always)]
fn flush_rows(stride: usize) -> usize {
    debug_assert!(stride / 8 <= PAIRS, "window wider than MAX_WINDOW");
    PAIRS / (stride / 8)
}

/// Lane masks: `MASK[16 - n..]` starts with `n` set lanes.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
static MASK: [i16; 32] = {
    let mut mask = [0i16; 32];
    let mut i = 0;
    while i < 16 {
        mask[i] = -1;
        i += 1;
    }
    mask
};

/// Two i16 weights in one i32 lane, `even` in the low half, as pairwise
/// multiply-adds of interleaved samples expect them.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn pair(even: i32, odd: i32) -> i32 {
    (even & 0xffff) | (odd << 16)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) use x86::Avx2;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
))]
pub(super) use x86::Sse2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    //! SSE2 and AVX2 kernels: `_mm_madd_epi16` / `_mm256_madd_epi16` over
    //! interleaved sample pairs, 8 or 16 pixels per step.

    use super::{
        GRADIENT_ROUND, GRADIENT_SHIFT, MASK, Ops, PIXEL_ROUND, PIXEL_SHIFT, Patch, Plane, Scalar,
        Source, Weights, flush_rows, pair,
    };
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    const PIXEL: i32 = PIXEL_SHIFT as i32;
    const GRADIENT: i32 = GRADIENT_SHIFT as i32;

    /// SSE2 kernels, part of the x86-64 baseline.
    #[cfg(target_feature = "sse2")]
    #[derive(Clone, Copy, Debug)]
    pub(in crate::lk) struct Sse2;

    /// AVX2 kernels. Only [`Avx2::detect`] creates one, after the CPU
    /// reported AVX2 support, which makes calling its kernels sound.
    #[derive(Clone, Copy, Debug)]
    pub(in crate::lk) struct Avx2(());

    impl Avx2 {
        pub(in crate::lk) fn detect() -> Option<Self> {
            is_x86_feature_detected!("avx2").then_some(Avx2(()))
        }
    }

    #[cfg(target_feature = "sse2")]
    impl Ops for Sse2 {
        #[inline(always)]
        fn build(
            self,
            src: Source<'_>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &mut Patch<'_>,
        ) -> [i64; 3] {
            if !src.pixels.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.build(src, x0, y0, w, patch);
            }
            // SAFETY: SSE2 is enabled at compile time.
            unsafe { build_sse2(src, x0, y0, w, patch) }
        }

        #[inline(always)]
        fn step(
            self,
            next: Plane<'_, u8>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &Patch<'_>,
        ) -> [i64; 2] {
            if !next.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.step(next, x0, y0, w, patch);
            }
            // SAFETY: SSE2 is enabled at compile time.
            unsafe { step_sse2(next, x0, y0, w, patch) }
        }

        #[inline(always)]
        fn error(
            self,
            next: Plane<'_, u8>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &Patch<'_>,
        ) -> i64 {
            if !next.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.error(next, x0, y0, w, patch);
            }
            // SAFETY: SSE2 is enabled at compile time.
            unsafe { error_sse2(next, x0, y0, w, patch) }
        }
    }

    impl Ops for Avx2 {
        #[inline(always)]
        fn build(
            self,
            src: Source<'_>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &mut Patch<'_>,
        ) -> [i64; 3] {
            if !src.pixels.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.build(src, x0, y0, w, patch);
            }
            // SAFETY: an `Avx2` exists only after runtime detection of AVX2.
            unsafe { build_avx2(src, x0, y0, w, patch) }
        }

        #[inline(always)]
        fn step(
            self,
            next: Plane<'_, u8>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &Patch<'_>,
        ) -> [i64; 2] {
            if !next.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.step(next, x0, y0, w, patch);
            }
            // SAFETY: an `Avx2` exists only after runtime detection of AVX2.
            unsafe { step_avx2(next, x0, y0, w, patch) }
        }

        #[inline(always)]
        fn error(
            self,
            next: Plane<'_, u8>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &Patch<'_>,
        ) -> i64 {
            if !next.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.error(next, x0, y0, w, patch);
            }
            // SAFETY: an `Avx2` exists only after runtime detection of AVX2.
            unsafe { error_avx2(next, x0, y0, w, patch) }
        }
    }

    /// 8 bytes of `row` from `i`, widened to i16 lanes.
    ///
    /// # Safety
    /// `i + 8 <= row.len()`.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn u8x8(row: &[u8], i: usize) -> __m128i {
        debug_assert!(i + 8 <= row.len());
        // SAFETY: the caller guarantees 8 readable bytes at `i`.
        let bytes = unsafe { _mm_loadl_epi64(row.as_ptr().add(i).cast()) };
        _mm_unpacklo_epi8(bytes, _mm_setzero_si128())
    }

    /// 8 values of `row` from `i`.
    ///
    /// # Safety
    /// `i + 8 <= row.len()`.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn i16x8(row: &[i16], i: usize) -> __m128i {
        debug_assert!(i + 8 <= row.len());
        // SAFETY: the caller guarantees 8 readable values at `i`.
        unsafe { _mm_loadu_si128(row.as_ptr().add(i).cast()) }
    }

    /// Stores 8 values into `row` from `i`.
    ///
    /// # Safety
    /// `i + 8 <= row.len()`.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn store8(row: &mut [i16], i: usize, value: __m128i) {
        debug_assert!(i + 8 <= row.len());
        // SAFETY: the caller guarantees 8 writable values at `i`.
        unsafe { _mm_storeu_si128(row.as_mut_ptr().add(i).cast(), value) }
    }

    /// The first `n` of 8 lanes set, `1 <= n <= 8`.
    #[inline]
    #[target_feature(enable = "sse2")]
    fn mask8(n: usize) -> __m128i {
        let lanes = &MASK[16 - n..16 - n + 8];
        // SAFETY: `lanes` holds exactly 8 values.
        unsafe { _mm_loadu_si128(lanes.as_ptr().cast()) }
    }

    /// The bilinear sums of 8 pixels from samples at `x` (`a0`, `b0`) and
    /// `x + 1` (`a1`, `b1`) of two rows, rounded and shifted to i16.
    #[inline]
    #[target_feature(enable = "sse2")]
    fn interpolate<const SHIFT: i32>(
        (a0, a1, b0, b1): (__m128i, __m128i, __m128i, __m128i),
        (wa, wb): (__m128i, __m128i),
        round: __m128i,
    ) -> __m128i {
        let lo = _mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi16(a0, a1), wa),
            _mm_madd_epi16(_mm_unpacklo_epi16(b0, b1), wb),
        );
        let hi = _mm_add_epi32(
            _mm_madd_epi16(_mm_unpackhi_epi16(a0, a1), wa),
            _mm_madd_epi16(_mm_unpackhi_epi16(b0, b1), wb),
        );
        _mm_packs_epi32(
            _mm_srai_epi32::<SHIFT>(_mm_add_epi32(lo, round)),
            _mm_srai_epi32::<SHIFT>(_mm_add_epi32(hi, round)),
        )
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    fn weights(w: Weights) -> (__m128i, __m128i) {
        (
            _mm_set1_epi32(pair(w.w00, w.w10)),
            _mm_set1_epi32(pair(w.w01, w.w11)),
        )
    }

    /// The sum of the four i32 lanes.
    #[inline]
    #[target_feature(enable = "sse2")]
    fn total(v: __m128i) -> i64 {
        let mut lanes = [0i32; 4];
        // SAFETY: `lanes` holds 16 writable bytes.
        unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), v) };
        lanes.iter().map(|&lane| lane as i64).sum()
    }

    /// The two source rows of a window row in the pixel and both gradient
    /// planes.
    type SourceRows<'a> = (
        &'a [u8],
        &'a [u8],
        &'a [i16],
        &'a [i16],
        &'a [i16],
        &'a [i16],
    );

    /// Patch values and gradients of 8 pixels at `i`, masked by `keep`, and
    /// their spatial products added to `acc`.
    ///
    /// # Safety
    /// `i + 9` elements of every source row and `i + 8` of every patch row.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn build_chunk(
        (pa, pb, xa, xb, ya, yb): SourceRows<'_>,
        (pixels, grad_x, grad_y): (&mut [i16], &mut [i16], &mut [i16]),
        i: usize,
        w: (__m128i, __m128i),
        keep: __m128i,
        acc: &mut [__m128i; 3],
    ) {
        let (pr, gr) = (_mm_set1_epi32(PIXEL_ROUND), _mm_set1_epi32(GRADIENT_ROUND));
        // SAFETY: the caller guarantees the extents.
        unsafe {
            let p = interpolate::<PIXEL>(
                (u8x8(pa, i), u8x8(pa, i + 1), u8x8(pb, i), u8x8(pb, i + 1)),
                w,
                pr,
            );
            let gx = interpolate::<GRADIENT>(
                (
                    i16x8(xa, i),
                    i16x8(xa, i + 1),
                    i16x8(xb, i),
                    i16x8(xb, i + 1),
                ),
                w,
                gr,
            );
            let gy = interpolate::<GRADIENT>(
                (
                    i16x8(ya, i),
                    i16x8(ya, i + 1),
                    i16x8(yb, i),
                    i16x8(yb, i + 1),
                ),
                w,
                gr,
            );
            let (gx, gy) = (_mm_and_si128(gx, keep), _mm_and_si128(gy, keep));
            store8(pixels, i, _mm_and_si128(p, keep));
            store8(grad_x, i, gx);
            store8(grad_y, i, gy);
            acc[0] = _mm_add_epi32(acc[0], _mm_madd_epi16(gx, gx));
            acc[1] = _mm_add_epi32(acc[1], _mm_madd_epi16(gx, gy));
            acc[2] = _mm_add_epi32(acc[2], _mm_madd_epi16(gy, gy));
        }
    }

    /// Residuals `I - J` of 8 pixels at `i`.
    ///
    /// # Safety
    /// `i + 9` elements of both rows and `i + 8` of `pixels`.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn residual8(
        (a, b): (&[u8], &[u8]),
        pixels: &[i16],
        i: usize,
        w: (__m128i, __m128i),
    ) -> __m128i {
        // SAFETY: the caller guarantees the extents.
        unsafe {
            let j = interpolate::<PIXEL>(
                (u8x8(a, i), u8x8(a, i + 1), u8x8(b, i), u8x8(b, i + 1)),
                w,
                _mm_set1_epi32(PIXEL_ROUND),
            );
            _mm_sub_epi16(i16x8(pixels, i), j)
        }
    }

    #[target_feature(enable = "sse2")]
    fn build_sse2(
        src: Source<'_>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &mut Patch<'_>,
    ) -> [i64; 3] {
        let (size, stride) = (patch.size, patch.stride);
        let (span, chunks) = (stride + 1, stride / 8);
        let (w, all, tail) = (weights(w), mask8(8), mask8(size - (chunks - 1) * 8));
        let rows_per_flush = flush_rows(stride);
        let mut sums = [0i64; 3];
        let mut acc = [_mm_setzero_si128(); 3];
        for j in 0..size {
            let (pa, pb) = src.pixels.rows(x0, y0, j, span);
            let (xa, xb) = src.grad_x.rows(x0, y0, j, span);
            let (ya, yb) = src.grad_y.rows(x0, y0, j, span);
            let row = j * stride..(j + 1) * stride;
            let pixels = &mut patch.pixels[row.clone()];
            let grad_x = &mut patch.grad_x[row.clone()];
            let grad_y = &mut patch.grad_y[row];
            for c in 0..chunks {
                let keep = if c + 1 == chunks { tail } else { all };
                // SAFETY: `8 c + 9 <= stride + 1 = span` source elements and
                // `8 c + 8 <= stride` patch elements per row.
                unsafe {
                    build_chunk(
                        (pa, pb, xa, xb, ya, yb),
                        (&mut *pixels, &mut *grad_x, &mut *grad_y),
                        8 * c,
                        w,
                        keep,
                        &mut acc,
                    );
                }
            }
            if (j + 1) % rows_per_flush == 0 || j + 1 == size {
                for (sum, lanes) in sums.iter_mut().zip(&mut acc) {
                    *sum += total(*lanes);
                    *lanes = _mm_setzero_si128();
                }
            }
        }
        sums
    }

    #[target_feature(enable = "sse2")]
    fn step_sse2(
        next: Plane<'_, u8>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &Patch<'_>,
    ) -> [i64; 2] {
        let (size, stride) = (patch.size, patch.stride);
        let (span, chunks) = (stride + 1, stride / 8);
        let w = weights(w);
        let rows_per_flush = flush_rows(stride);
        let mut sums = [0i64; 2];
        let (mut bx, mut by) = (_mm_setzero_si128(), _mm_setzero_si128());
        for j in 0..size {
            let rows = next.rows(x0, y0, j, span);
            let row = j * stride..(j + 1) * stride;
            let (pixels, grad_x, grad_y) = (
                &patch.pixels[row.clone()],
                &patch.grad_x[row.clone()],
                &patch.grad_y[row],
            );
            for c in 0..chunks {
                let i = 8 * c;
                // SAFETY: `i + 9 <= span` and `i + 8 <= stride`. Lanes past
                // `size` have zero gradients and add nothing.
                unsafe {
                    let d = residual8(rows, pixels, i, w);
                    bx = _mm_add_epi32(bx, _mm_madd_epi16(d, i16x8(grad_x, i)));
                    by = _mm_add_epi32(by, _mm_madd_epi16(d, i16x8(grad_y, i)));
                }
            }
            if (j + 1) % rows_per_flush == 0 || j + 1 == size {
                sums[0] += total(bx);
                sums[1] += total(by);
                (bx, by) = (_mm_setzero_si128(), _mm_setzero_si128());
            }
        }
        sums
    }

    #[target_feature(enable = "sse2")]
    fn error_sse2(next: Plane<'_, u8>, x0: usize, y0: usize, w: Weights, patch: &Patch<'_>) -> i64 {
        let (size, stride) = (patch.size, patch.stride);
        let (span, chunks) = (stride + 1, stride / 8);
        let (w, all, tail) = (weights(w), mask8(8), mask8(size - (chunks - 1) * 8));
        let ones = _mm_set1_epi16(1);
        let mut acc = _mm_setzero_si128();
        for j in 0..size {
            let rows = next.rows(x0, y0, j, span);
            let pixels = &patch.pixels[j * stride..(j + 1) * stride];
            for c in 0..chunks {
                let keep = if c + 1 == chunks { tail } else { all };
                // SAFETY: `8 c + 9 <= span` and `8 c + 8 <= stride`.
                let d = unsafe { residual8(rows, pixels, 8 * c, w) };
                acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_and_si128(abs8(d), keep), ones));
            }
        }
        total(acc)
    }

    /// 16 bytes of `row` from `i`, widened to i16 lanes.
    ///
    /// # Safety
    /// AVX2 and `i + 16 <= row.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn u8x16(row: &[u8], i: usize) -> __m256i {
        debug_assert!(i + 16 <= row.len());
        // SAFETY: the caller guarantees 16 readable bytes at `i`.
        _mm256_cvtepu8_epi16(unsafe { _mm_loadu_si128(row.as_ptr().add(i).cast()) })
    }

    /// 16 values of `row` from `i`.
    ///
    /// # Safety
    /// AVX2 and `i + 16 <= row.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn i16x16(row: &[i16], i: usize) -> __m256i {
        debug_assert!(i + 16 <= row.len());
        // SAFETY: the caller guarantees 16 readable values at `i`.
        unsafe { _mm256_loadu_si256(row.as_ptr().add(i).cast()) }
    }

    /// Stores 16 values into `row` from `i`.
    ///
    /// # Safety
    /// AVX2 and `i + 16 <= row.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store16(row: &mut [i16], i: usize, value: __m256i) {
        debug_assert!(i + 16 <= row.len());
        // SAFETY: the caller guarantees 16 writable values at `i`.
        unsafe { _mm256_storeu_si256(row.as_mut_ptr().add(i).cast(), value) }
    }

    /// The first `n` of 16 lanes set, `1 <= n <= 16`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn mask16(n: usize) -> __m256i {
        let lanes = &MASK[16 - n..32 - n];
        // SAFETY: `lanes` holds exactly 16 values.
        unsafe { _mm256_loadu_si256(lanes.as_ptr().cast()) }
    }

    /// [`interpolate`] for 16 pixels. The 256-bit unpacks, multiply-adds and
    /// packs all work within 128-bit halves, which keeps the pixel order.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn interpolate16<const SHIFT: i32>(
        (a0, a1, b0, b1): (__m256i, __m256i, __m256i, __m256i),
        (wa, wb): (__m256i, __m256i),
        round: __m256i,
    ) -> __m256i {
        let lo = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_unpacklo_epi16(a0, a1), wa),
            _mm256_madd_epi16(_mm256_unpacklo_epi16(b0, b1), wb),
        );
        let hi = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_unpackhi_epi16(a0, a1), wa),
            _mm256_madd_epi16(_mm256_unpackhi_epi16(b0, b1), wb),
        );
        _mm256_packs_epi32(
            _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(lo, round)),
            _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(hi, round)),
        )
    }

    /// The sum of the eight i32 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn total16(v: __m256i) -> i64 {
        total(_mm256_castsi256_si128(v)) + total(_mm256_extracti128_si256::<1>(v))
    }

    /// Residuals `I - J` of 16 pixels at `i`.
    ///
    /// # Safety
    /// AVX2, `i + 17` elements of both rows and `i + 16` of `pixels`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn residual16(
        (a, b): (&[u8], &[u8]),
        pixels: &[i16],
        i: usize,
        w: (__m256i, __m256i),
    ) -> __m256i {
        // SAFETY: the caller guarantees the extents.
        unsafe {
            let j = interpolate16::<PIXEL>(
                (u8x16(a, i), u8x16(a, i + 1), u8x16(b, i), u8x16(b, i + 1)),
                w,
                _mm256_set1_epi32(PIXEL_ROUND),
            );
            _mm256_sub_epi16(i16x16(pixels, i), j)
        }
    }

    /// AVX2 weights: the SSE2 pairs in both halves.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn weights16(w: Weights) -> (__m256i, __m256i) {
        (
            _mm256_set1_epi32(pair(w.w00, w.w10)),
            _mm256_set1_epi32(pair(w.w01, w.w11)),
        )
    }

    // The AVX2 kernels step through a row in 16 pixels while more than 8
    // remain (`i + 8 < size`) and finish with at most one 8-pixel SSE2 step
    // if any remain. More than 8 remaining
    // pixels round up to at least 16 within the stride, so every 16-pixel
    // step stays inside the patch row and reads at most `stride + 1` source
    // elements, like the 8-pixel steps.

    #[target_feature(enable = "avx2")]
    fn build_avx2(
        src: Source<'_>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &mut Patch<'_>,
    ) -> [i64; 3] {
        let (size, stride) = (patch.size, patch.stride);
        let span = stride + 1;
        let (w8, w16) = (weights(w), weights16(w));
        let (pr, gr) = (
            _mm256_set1_epi32(PIXEL_ROUND),
            _mm256_set1_epi32(GRADIENT_ROUND),
        );
        let rows_per_flush = flush_rows(stride);
        let mut sums = [0i64; 3];
        let mut acc8 = [_mm_setzero_si128(); 3];
        let mut acc16 = [_mm256_setzero_si256(); 3];
        for j in 0..size {
            let (pa, pb) = src.pixels.rows(x0, y0, j, span);
            let (xa, xb) = src.grad_x.rows(x0, y0, j, span);
            let (ya, yb) = src.grad_y.rows(x0, y0, j, span);
            let row = j * stride..(j + 1) * stride;
            let pixels = &mut patch.pixels[row.clone()];
            let grad_x = &mut patch.grad_x[row.clone()];
            let grad_y = &mut patch.grad_y[row];
            let mut i = 0;
            while i + 8 < size {
                let keep = mask16((size - i).min(16));
                // SAFETY: `i + 17 <= span` source elements and `i + 16 <=
                // stride` patch elements, see above.
                unsafe {
                    let p = interpolate16::<PIXEL>(
                        (
                            u8x16(pa, i),
                            u8x16(pa, i + 1),
                            u8x16(pb, i),
                            u8x16(pb, i + 1),
                        ),
                        w16,
                        pr,
                    );
                    let gx = interpolate16::<GRADIENT>(
                        (
                            i16x16(xa, i),
                            i16x16(xa, i + 1),
                            i16x16(xb, i),
                            i16x16(xb, i + 1),
                        ),
                        w16,
                        gr,
                    );
                    let gy = interpolate16::<GRADIENT>(
                        (
                            i16x16(ya, i),
                            i16x16(ya, i + 1),
                            i16x16(yb, i),
                            i16x16(yb, i + 1),
                        ),
                        w16,
                        gr,
                    );
                    let (gx, gy) = (_mm256_and_si256(gx, keep), _mm256_and_si256(gy, keep));
                    store16(pixels, i, _mm256_and_si256(p, keep));
                    store16(grad_x, i, gx);
                    store16(grad_y, i, gy);
                    acc16[0] = _mm256_add_epi32(acc16[0], _mm256_madd_epi16(gx, gx));
                    acc16[1] = _mm256_add_epi32(acc16[1], _mm256_madd_epi16(gx, gy));
                    acc16[2] = _mm256_add_epi32(acc16[2], _mm256_madd_epi16(gy, gy));
                }
                i += 16;
            }
            if i < size {
                // SAFETY: `i + 8 <= stride`.
                unsafe {
                    build_chunk(
                        (pa, pb, xa, xb, ya, yb),
                        (&mut *pixels, &mut *grad_x, &mut *grad_y),
                        i,
                        w8,
                        mask8(size - i),
                        &mut acc8,
                    );
                }
            }
            if (j + 1) % rows_per_flush == 0 || j + 1 == size {
                for k in 0..3 {
                    sums[k] += total(acc8[k]) + total16(acc16[k]);
                    acc8[k] = _mm_setzero_si128();
                    acc16[k] = _mm256_setzero_si256();
                }
            }
        }
        sums
    }

    #[target_feature(enable = "avx2")]
    fn step_avx2(
        next: Plane<'_, u8>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &Patch<'_>,
    ) -> [i64; 2] {
        let (size, stride) = (patch.size, patch.stride);
        let span = stride + 1;
        let (w8, w16) = (weights(w), weights16(w));
        let rows_per_flush = flush_rows(stride);
        let mut sums = [0i64; 2];
        let (mut bx8, mut by8) = (_mm_setzero_si128(), _mm_setzero_si128());
        let (mut bx16, mut by16) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for j in 0..size {
            let rows = next.rows(x0, y0, j, span);
            let row = j * stride..(j + 1) * stride;
            let (pixels, grad_x, grad_y) = (
                &patch.pixels[row.clone()],
                &patch.grad_x[row.clone()],
                &patch.grad_y[row],
            );
            let mut i = 0;
            while i + 8 < size {
                // SAFETY: `i + 17 <= span` and `i + 16 <= stride`. Lanes past
                // `size` have zero gradients and add nothing.
                unsafe {
                    let d = residual16(rows, pixels, i, w16);
                    bx16 = _mm256_add_epi32(bx16, _mm256_madd_epi16(d, i16x16(grad_x, i)));
                    by16 = _mm256_add_epi32(by16, _mm256_madd_epi16(d, i16x16(grad_y, i)));
                }
                i += 16;
            }
            if i < size {
                // SAFETY: `i + 8 <= stride`.
                unsafe {
                    let d = residual8(rows, pixels, i, w8);
                    bx8 = _mm_add_epi32(bx8, _mm_madd_epi16(d, i16x8(grad_x, i)));
                    by8 = _mm_add_epi32(by8, _mm_madd_epi16(d, i16x8(grad_y, i)));
                }
            }
            if (j + 1) % rows_per_flush == 0 || j + 1 == size {
                sums[0] += total(bx8) + total16(bx16);
                sums[1] += total(by8) + total16(by16);
                (bx8, by8) = (_mm_setzero_si128(), _mm_setzero_si128());
                (bx16, by16) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            }
        }
        sums
    }

    #[target_feature(enable = "avx2")]
    fn error_avx2(next: Plane<'_, u8>, x0: usize, y0: usize, w: Weights, patch: &Patch<'_>) -> i64 {
        let (size, stride) = (patch.size, patch.stride);
        let span = stride + 1;
        let (w8, w16) = (weights(w), weights16(w));
        let (ones8, ones16) = (_mm_set1_epi16(1), _mm256_set1_epi16(1));
        let mut acc8 = _mm_setzero_si128();
        let mut acc16 = _mm256_setzero_si256();
        for j in 0..size {
            let rows = next.rows(x0, y0, j, span);
            let pixels = &patch.pixels[j * stride..(j + 1) * stride];
            let mut i = 0;
            while i + 8 < size {
                // SAFETY: `i + 17 <= span` and `i + 16 <= stride`.
                let d = unsafe { residual16(rows, pixels, i, w16) };
                let d = _mm256_and_si256(_mm256_abs_epi16(d), mask16((size - i).min(16)));
                acc16 = _mm256_add_epi32(acc16, _mm256_madd_epi16(d, ones16));
                i += 16;
            }
            if i < size {
                // SAFETY: `i + 9 <= span` and `i + 8 <= stride`.
                let d = unsafe { residual8(rows, pixels, i, w8) };
                let d = _mm_and_si128(abs8(d), mask8(size - i));
                acc8 = _mm_add_epi32(acc8, _mm_madd_epi16(d, ones8));
            }
        }
        total(acc8) + total16(acc16)
    }

    /// Lane-wise |v| for |v| < 2^15 with SSE2 only.
    #[inline]
    #[target_feature(enable = "sse2")]
    fn abs8(v: __m128i) -> __m128i {
        _mm_max_epi16(v, _mm_sub_epi16(_mm_setzero_si128(), v))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) use wasm::Simd128;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    //! WebAssembly `simd128` kernels, 8 pixels per step, with the widening
    //! multiplies `i32x4.extmul_{low,high}_i16x8_s` and the weights in all
    //! lanes. On arm64, where V8 lowers them to `smull`/`smlal`, they beat
    //! `i32x4.dot_i16x8_s` over interleaved sample pairs (a `smull`,
    //! `smull2`, `addp` triple plus the interleaving shuffles) by 10-16% per
    //! frame step on a Cortex-A78 phone; on x86-64 V8, which lowers the dot
    //! product to `pmaddwd` but a widening multiply to three instructions,
    //! the dot product variant was 37% faster. The phone is the deployment
    //! target. Both compute the same integers.

    use super::{
        GRADIENT_ROUND, GRADIENT_SHIFT, MASK, Ops, PIXEL_ROUND, PIXEL_SHIFT, Patch, Plane, Scalar,
        Source, Weights, flush_rows,
    };
    use core::arch::wasm32::*;

    /// `simd128` kernels, enabled at compile time.
    #[derive(Clone, Copy, Debug)]
    pub(in crate::lk) struct Simd128;

    impl Ops for Simd128 {
        #[inline(always)]
        fn build(
            self,
            src: Source<'_>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &mut Patch<'_>,
        ) -> [i64; 3] {
            if !src.pixels.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.build(src, x0, y0, w, patch);
            }
            // `simd128` is enabled at compile time, so the call is safe.
            build_simd128(src, x0, y0, w, patch)
        }

        #[inline(always)]
        fn step(
            self,
            next: Plane<'_, u8>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &Patch<'_>,
        ) -> [i64; 2] {
            if !next.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.step(next, x0, y0, w, patch);
            }
            // `simd128` is enabled at compile time, so the call is safe.
            step_simd128(next, x0, y0, w, patch)
        }

        #[inline(always)]
        fn error(
            self,
            next: Plane<'_, u8>,
            x0: usize,
            y0: usize,
            w: Weights,
            patch: &Patch<'_>,
        ) -> i64 {
            if !next.holds(x0, y0, patch.size, patch.stride + 1) {
                return Scalar.error(next, x0, y0, w, patch);
            }
            // `simd128` is enabled at compile time, so the call is safe.
            error_simd128(next, x0, y0, w, patch)
        }
    }

    /// 8 bytes of `row` from `i`, widened to i16 lanes.
    ///
    /// # Safety
    /// `i + 8 <= row.len()`.
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn u8x8(row: &[u8], i: usize) -> v128 {
        debug_assert!(i + 8 <= row.len());
        // SAFETY: the caller guarantees 8 readable bytes at `i`.
        unsafe { u16x8_load_extend_u8x8(row.as_ptr().add(i)) }
    }

    /// 8 values of `row` from `i`.
    ///
    /// # Safety
    /// `i + 8 <= row.len()`.
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn i16x8(row: &[i16], i: usize) -> v128 {
        debug_assert!(i + 8 <= row.len());
        // SAFETY: the caller guarantees 8 readable values at `i`.
        unsafe { v128_load(row.as_ptr().add(i).cast()) }
    }

    /// Stores 8 values into `row` from `i`.
    ///
    /// # Safety
    /// `i + 8 <= row.len()`.
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn store8(row: &mut [i16], i: usize, value: v128) {
        debug_assert!(i + 8 <= row.len());
        // SAFETY: the caller guarantees 8 writable values at `i`.
        unsafe { v128_store(row.as_mut_ptr().add(i).cast(), value) }
    }

    /// The first `n` of 8 lanes set, `1 <= n <= 8`.
    #[inline]
    #[target_feature(enable = "simd128")]
    fn mask8(n: usize) -> v128 {
        let lanes = &MASK[16 - n..16 - n + 8];
        // SAFETY: `lanes` holds exactly 8 values.
        unsafe { v128_load(lanes.as_ptr().cast()) }
    }

    /// The bilinear sums of 8 pixels from samples at `x` (`a0`, `b0`) and
    /// `x + 1` (`a1`, `b1`) of two rows, rounded and shifted to i16.
    #[inline]
    #[target_feature(enable = "simd128")]
    fn interpolate<const SHIFT: u32>(
        (a0, a1, b0, b1): (v128, v128, v128, v128),
        w: Lanes,
        round: v128,
    ) -> v128 {
        let lo = i32x4_add(
            i32x4_add(
                i32x4_add(round, i32x4_extmul_low_i16x8(a0, w.w00)),
                i32x4_extmul_low_i16x8(a1, w.w10),
            ),
            i32x4_add(
                i32x4_extmul_low_i16x8(b0, w.w01),
                i32x4_extmul_low_i16x8(b1, w.w11),
            ),
        );
        let hi = i32x4_add(
            i32x4_add(
                i32x4_add(round, i32x4_extmul_high_i16x8(a0, w.w00)),
                i32x4_extmul_high_i16x8(a1, w.w10),
            ),
            i32x4_add(
                i32x4_extmul_high_i16x8(b0, w.w01),
                i32x4_extmul_high_i16x8(b1, w.w11),
            ),
        );
        i16x8_narrow_i32x4(i32x4_shr(lo, SHIFT), i32x4_shr(hi, SHIFT))
    }

    /// The four weights, each in all eight i16 lanes.
    #[derive(Clone, Copy)]
    struct Lanes {
        w00: v128,
        w10: v128,
        w01: v128,
        w11: v128,
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    fn weights(w: Weights) -> Lanes {
        Lanes {
            w00: i16x8_splat(w.w00 as i16),
            w10: i16x8_splat(w.w10 as i16),
            w01: i16x8_splat(w.w01 as i16),
            w11: i16x8_splat(w.w11 as i16),
        }
    }

    /// `acc + a * b` over all 8 lanes, two products per i32 lane.
    #[inline]
    #[target_feature(enable = "simd128")]
    fn multiply_add(acc: v128, a: v128, b: v128) -> v128 {
        i32x4_add(
            i32x4_add(acc, i32x4_extmul_low_i16x8(a, b)),
            i32x4_extmul_high_i16x8(a, b),
        )
    }

    /// The sum of the four i32 lanes.
    #[inline]
    #[target_feature(enable = "simd128")]
    fn total(v: v128) -> i64 {
        i32x4_extract_lane::<0>(v) as i64
            + i32x4_extract_lane::<1>(v) as i64
            + i32x4_extract_lane::<2>(v) as i64
            + i32x4_extract_lane::<3>(v) as i64
    }

    /// Residuals `I - J` of 8 pixels at `i`.
    ///
    /// # Safety
    /// `i + 9` elements of both rows and `i + 8` of `pixels`.
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn residual8((a, b): (&[u8], &[u8]), pixels: &[i16], i: usize, w: Lanes) -> v128 {
        // SAFETY: the caller guarantees the extents.
        unsafe {
            let j = interpolate::<PIXEL_SHIFT>(
                (u8x8(a, i), u8x8(a, i + 1), u8x8(b, i), u8x8(b, i + 1)),
                w,
                i32x4_splat(PIXEL_ROUND),
            );
            i16x8_sub(i16x8(pixels, i), j)
        }
    }

    #[target_feature(enable = "simd128")]
    fn build_simd128(
        src: Source<'_>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &mut Patch<'_>,
    ) -> [i64; 3] {
        let (size, stride) = (patch.size, patch.stride);
        let (span, chunks) = (stride + 1, stride / 8);
        let (w, all, tail) = (weights(w), mask8(8), mask8(size - (chunks - 1) * 8));
        let (pr, gr) = (i32x4_splat(PIXEL_ROUND), i32x4_splat(GRADIENT_ROUND));
        let rows_per_flush = flush_rows(stride);
        let mut sums = [0i64; 3];
        let mut acc = [i32x4_splat(0); 3];
        for j in 0..size {
            let (pa, pb) = src.pixels.rows(x0, y0, j, span);
            let (xa, xb) = src.grad_x.rows(x0, y0, j, span);
            let (ya, yb) = src.grad_y.rows(x0, y0, j, span);
            let row = j * stride..(j + 1) * stride;
            let pixels = &mut patch.pixels[row.clone()];
            let grad_x = &mut patch.grad_x[row.clone()];
            let grad_y = &mut patch.grad_y[row];
            for c in 0..chunks {
                let (i, keep) = (8 * c, if c + 1 == chunks { tail } else { all });
                // SAFETY: `i + 9 <= stride + 1 = span` source elements and
                // `i + 8 <= stride` patch elements per row.
                unsafe {
                    let p = interpolate::<PIXEL_SHIFT>(
                        (u8x8(pa, i), u8x8(pa, i + 1), u8x8(pb, i), u8x8(pb, i + 1)),
                        w,
                        pr,
                    );
                    let gx = interpolate::<GRADIENT_SHIFT>(
                        (
                            i16x8(xa, i),
                            i16x8(xa, i + 1),
                            i16x8(xb, i),
                            i16x8(xb, i + 1),
                        ),
                        w,
                        gr,
                    );
                    let gy = interpolate::<GRADIENT_SHIFT>(
                        (
                            i16x8(ya, i),
                            i16x8(ya, i + 1),
                            i16x8(yb, i),
                            i16x8(yb, i + 1),
                        ),
                        w,
                        gr,
                    );
                    let (gx, gy) = (v128_and(gx, keep), v128_and(gy, keep));
                    store8(pixels, i, v128_and(p, keep));
                    store8(grad_x, i, gx);
                    store8(grad_y, i, gy);
                    acc[0] = multiply_add(acc[0], gx, gx);
                    acc[1] = multiply_add(acc[1], gx, gy);
                    acc[2] = multiply_add(acc[2], gy, gy);
                }
            }
            if (j + 1) % rows_per_flush == 0 || j + 1 == size {
                for (sum, lanes) in sums.iter_mut().zip(&mut acc) {
                    *sum += total(*lanes);
                    *lanes = i32x4_splat(0);
                }
            }
        }
        sums
    }

    #[target_feature(enable = "simd128")]
    fn step_simd128(
        next: Plane<'_, u8>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &Patch<'_>,
    ) -> [i64; 2] {
        let (size, stride) = (patch.size, patch.stride);
        let (span, chunks) = (stride + 1, stride / 8);
        let w = weights(w);
        let rows_per_flush = flush_rows(stride);
        let mut sums = [0i64; 2];
        let (mut bx, mut by) = (i32x4_splat(0), i32x4_splat(0));
        for j in 0..size {
            let rows = next.rows(x0, y0, j, span);
            let row = j * stride..(j + 1) * stride;
            let (pixels, grad_x, grad_y) = (
                &patch.pixels[row.clone()],
                &patch.grad_x[row.clone()],
                &patch.grad_y[row],
            );
            for c in 0..chunks {
                let i = 8 * c;
                // SAFETY: `i + 9 <= span` and `i + 8 <= stride`. Lanes past
                // `size` have zero gradients and add nothing.
                unsafe {
                    let d = residual8(rows, pixels, i, w);
                    bx = multiply_add(bx, d, i16x8(grad_x, i));
                    by = multiply_add(by, d, i16x8(grad_y, i));
                }
            }
            if (j + 1) % rows_per_flush == 0 || j + 1 == size {
                sums[0] += total(bx);
                sums[1] += total(by);
                (bx, by) = (i32x4_splat(0), i32x4_splat(0));
            }
        }
        sums
    }

    #[target_feature(enable = "simd128")]
    fn error_simd128(
        next: Plane<'_, u8>,
        x0: usize,
        y0: usize,
        w: Weights,
        patch: &Patch<'_>,
    ) -> i64 {
        let (size, stride) = (patch.size, patch.stride);
        let (span, chunks) = (stride + 1, stride / 8);
        let (w, all, tail) = (weights(w), mask8(8), mask8(size - (chunks - 1) * 8));
        let mut acc = i32x4_splat(0);
        for j in 0..size {
            let rows = next.rows(x0, y0, j, span);
            let pixels = &patch.pixels[j * stride..(j + 1) * stride];
            for c in 0..chunks {
                let keep = if c + 1 == chunks { tail } else { all };
                // SAFETY: `8 c + 9 <= span` and `8 c + 8 <= stride`.
                let d = unsafe { residual8(rows, pixels, 8 * c, w) };
                acc = i32x4_add(
                    acc,
                    i32x4_extadd_pairwise_i16x8(v128_and(i16x8_abs(d), keep)),
                );
            }
        }
        total(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::fast_gradients::compute_gradients_into;
    use image::{GrayImage, Luma};

    /// Everything one kernel computes for one window pair.
    #[derive(Debug, PartialEq)]
    struct Outcome {
        spatial: [i64; 3],
        patch: [Vec<i16>; 3],
        residual: [i64; 2],
        error: i64,
    }

    /// A previous frame with its gradients, a next frame and two windows.
    struct Case {
        prev: GrayImage,
        grad_x: Vec<i16>,
        grad_y: Vec<i16>,
        next: GrayImage,
        size: usize,
        prev_window: Window,
        next_window: Window,
    }

    impl Case {
        fn new(
            prev: GrayImage,
            next: GrayImage,
            size: usize,
            at: (f32, f32),
            to: (f32, f32),
        ) -> Self {
            let (w, h) = prev.dimensions();
            let n = (w * h) as usize;
            let (mut grad_x, mut grad_y) = (vec![0; n], vec![0; n]);
            compute_gradients_into(&prev, &mut grad_x, &mut grad_y);
            let window = |(x, y): (f32, f32)| {
                Window::new(x, y, size / 2, w as usize, h as usize)
                    .expect("window inside the frame")
            };
            Case {
                prev_window: window(at),
                next_window: window(to),
                prev,
                grad_x,
                grad_y,
                next,
                size,
            }
        }

        fn run<K: Ops>(&self, ops: K) -> Outcome {
            let width = self.prev.width() as usize;
            let stride = stride(self.size);
            // Stale values that every kernel must overwrite, padding included.
            let mut buffers = [(); 3].map(|_| vec![0x5a5a_i16; self.size * stride]);
            let [pixels, grad_x, grad_y] = &mut buffers;
            let mut patch = Patch {
                size: self.size,
                stride,
                pixels,
                grad_x,
                grad_y,
            };
            let source = Source {
                pixels: Plane {
                    data: self.prev.as_raw(),
                    width,
                },
                grad_x: Plane {
                    data: &self.grad_x,
                    width,
                },
                grad_y: Plane {
                    data: &self.grad_y,
                    width,
                },
            };
            let next = Plane {
                data: self.next.as_raw(),
                width,
            };
            let (pw, nw) = (&self.prev_window, &self.next_window);
            let spatial = ops.build(source, pw.x0, pw.y0, Weights::of(pw), &mut patch);
            let residual = ops.step(next, nw.x0, nw.y0, Weights::of(nw), &patch);
            let error = ops.error(next, nw.x0, nw.y0, Weights::of(nw), &patch);
            Outcome {
                spatial,
                patch: buffers,
                residual,
                error,
            }
        }

        /// The outcome of every kernel this CPU runs.
        fn outcomes(&self) -> Vec<(&'static str, Outcome)> {
            #[allow(unused_mut)]
            let mut outcomes = vec![("scalar", self.run(Scalar))];
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sse2"
            ))]
            outcomes.push(("sse2", self.run(Sse2)));
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(avx2) = Avx2::detect() {
                outcomes.push(("avx2", self.run(avx2)));
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            outcomes.push(("simd128", self.run(Simd128)));
            outcomes
        }

        /// Asserts that every kernel computes the scalar outcome.
        fn assert_kernels_agree(&self) -> Outcome {
            let mut outcomes = self.outcomes().into_iter();
            let (_, scalar) = outcomes.next().unwrap();
            for (name, outcome) in outcomes {
                assert_eq!(
                    outcome,
                    scalar,
                    "{name} differs from scalar: size {} at {:?} -> {:?}",
                    self.size,
                    (self.prev_window.x0, self.prev_window.y0),
                    (self.next_window.x0, self.next_window.y0)
                );
            }
            scalar
        }
    }

    fn noise(w: u32, h: u32, seed: u32) -> GrayImage {
        GrayImage::from_fn(w, h, |x, y| {
            let mut s = (y * w + x).wrapping_mul(2654435761) ^ seed.wrapping_mul(0x9e3779b9);
            s ^= s >> 15;
            s = s.wrapping_mul(0x85ebca6b);
            Luma([(s >> 7) as u8])
        })
    }

    #[test]
    fn weights_sum_to_one_and_dip_at_most_one_below_zero() {
        let mut negative = 0;
        for i in 0..=400 {
            for j in 0..=400 {
                let (fx, fy) = (i as f32 / 401.0, j as f32 / 401.0 * 0.03);
                let win = Window::new(5.0 + fx, 5.0 + fy, 2, 20, 20).unwrap();
                let w = Weights::of(&win);
                assert_eq!(w.w00 + w.w10 + w.w01 + w.w11, WEIGHT_ONE);
                assert!(
                    w.w00 >= 0 && w.w10 >= 0 && w.w01 >= 0 && w.w11 >= -1,
                    "{w:?}"
                );
                negative += (w.w11 < 0) as usize;
            }
        }
        // The case the bounds allow for does occur.
        assert!(negative > 0);
    }

    #[test]
    fn scalar_kernel_samples_like_exact_bilinear_interpolation() {
        let (prev, next) = (noise(40, 30, 1), noise(40, 30, 2));
        for (at, size) in [((17.0, 12.0), 7), ((17.3, 14.6), 9), ((20.81, 11.07), 13)] {
            let case = Case::new(prev.clone(), next.clone(), size, at, at);
            let outcome = case.run(Scalar);
            let stride = stride(size);
            let r = (size / 2) as f32;
            for j in 0..size {
                for i in 0..size {
                    let (x, y) = (at.0 - r + i as f32, at.1 - r + j as f32);
                    let (x0, y0) = (x.floor(), y.floor());
                    let (fx, fy) = ((x - x0) as f64, (y - y0) as f64);
                    let p =
                        |dx: u32, dy: u32| prev.get_pixel(x0 as u32 + dx, y0 as u32 + dy)[0] as f64;
                    let exact = p(0, 0) * (1.0 - fx) * (1.0 - fy)
                        + p(1, 0) * fx * (1.0 - fy)
                        + p(0, 1) * (1.0 - fx) * fy
                        + p(1, 1) * fx * fy;
                    let got = outcome.patch[0][j * stride + i] as f64 / 32.0;
                    // Rounding to 1/32 grey level and 14-bit weights.
                    assert!(
                        (got - exact).abs() <= 1.0 / 64.0 + 255.0 * 3.0 / 16384.0,
                        "{got} vs {exact}"
                    );
                    if fx == 0.0 && fy == 0.0 {
                        assert_eq!(got, exact);
                    }
                }
                for plane in &outcome.patch {
                    assert!(
                        plane[j * stride + size..(j + 1) * stride]
                            .iter()
                            .all(|&v| v == 0)
                    );
                }
            }
        }
    }

    #[test]
    fn every_kernel_computes_the_scalar_integers() {
        // Odd frame widths, every window size up to 33, fractional and
        // integer positions, and windows whose vector over-read would leave
        // the frame (bottom-right corner), which fall back to the scalar path.
        let (w, h) = (67u32, 53u32);
        let (prev, next) = (noise(w, h, 3), noise(w, h, 4));
        for size in (1..=33).step_by(2) {
            let r = (size / 2) as f32;
            let far = (w as f32 - r - 1.001, h as f32 - r - 1.001);
            let positions = [
                (r, r),
                (r + 0.5, r + 0.25),
                (r + 3.004, r + 7.004),
                (30.37, 26.61),
                (far.0, far.1),
                (far.0 - 0.6, far.1),
                (far.0, far.1 - 2.3),
            ];
            for &at in &positions {
                for &to in &positions {
                    Case::new(prev.clone(), next.clone(), size, at, to).assert_kernels_agree();
                }
            }
        }
    }

    #[test]
    fn adversarial_content_does_not_overflow_any_lane() {
        // Stripes 0, 0, 255, 255 give the largest Scharr response, 16 * 255,
        // at every pixel. The next frame shifted by one stripe pixel gives
        // the largest residual, 32 * 255, on every other pixel, with the
        // sign of the gradient there.
        let stripes = |shift: u32, vertical: bool| {
            GrayImage::from_fn(300, 300, move |x, y| {
                let t = if vertical { x } else { y } + shift;
                Luma([if t % 4 >= 2 { 255 } else { 0 }])
            })
        };
        for vertical in [true, false] {
            let (prev, next) = (stripes(0, vertical), stripes(1, vertical));
            for size in [1usize, 13, 21, 63, 127] {
                let at = (150.0, 150.0);
                let outcome =
                    Case::new(prev.clone(), next.clone(), size, at, at).assert_kernels_agree();
                // The spatial sum is the bound itself: every sample at 8160.
                let n = (size * size) as i64;
                let along = if vertical { 0 } else { 2 };
                assert_eq!(outcome.spatial[along], n * GRADIENT_MAX * GRADIENT_MAX);
                if size >= 21 {
                    // Beyond i32, so the lanes must have been flushed in time.
                    assert!(outcome.spatial[along] > i32::MAX as i64);
                }
                // Pixels at odd positions along the stripes differ by 255,
                // against a gradient of the opposite sign.
                let start = 150 - size / 2;
                let odd = (start..start + size).filter(|t| t % 2 == 1).count() as i64;
                let residual = outcome.residual[if vertical { 0 } else { 1 }];
                assert_eq!(residual, -(odd * size as i64) * PIXEL_MAX * GRADIENT_MAX);
            }
        }
    }
}
