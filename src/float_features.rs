use crate::features::{
    DetectorCapacity, GridLayout, Planes, Selection, local_maxima, min_eigenvalue_scores,
    quality_floor, sort_by_score,
};
use crate::float_image::FloatImageView;
use crate::utils::float_gradients::scharr_f32_into;

/// Corner score threshold of the f32 detector.
///
/// Scores are minimum eigenvalues of the mean 3x3 structure tensor of Scharr
/// gradients normalized by 32, in (input unit / pixel)^2: (°C / px)^2 for a
/// thermal image in °C. They scale with the square of the image contrast and
/// do not depend on a constant offset.
///
/// A local maximum is kept when its score is positive and at least
/// `max(absolute, relative * strongest)`, where `strongest` is the largest
/// local-maximum score in the whole image. The reference is global for grid
/// detection as well, not per cell. The relative part is disabled (zero)
/// unless set with [`with_relative`](Self::with_relative).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreThreshold {
    absolute: f32,
    relative: f32,
}

impl ScoreThreshold {
    /// Keeps local maxima scoring at least `min_score`, in
    /// (input unit / pixel)^2.
    ///
    /// # Panics
    /// Panics if `min_score` is negative, NaN or infinite.
    pub fn absolute(min_score: f32) -> Self {
        assert!(
            min_score.is_finite() && min_score >= 0.0,
            "absolute score threshold must be finite and non-negative, got {min_score}"
        );
        Self {
            absolute: min_score,
            relative: 0.0,
        }
    }

    /// Additionally requires at least `quality_level` times the strongest
    /// local maximum's score, as the 8-bit detector's `quality_level` does.
    ///
    /// # Panics
    /// Panics unless `0 <= quality_level <= 1`.
    pub fn with_relative(self, quality_level: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&quality_level),
            "relative score threshold must be within 0..=1, got {quality_level}"
        );
        Self {
            relative: quality_level,
            ..self
        }
    }

    /// The absolute part, in (input unit / pixel)^2.
    pub fn absolute_part(&self) -> f32 {
        self.absolute
    }

    /// The relative part; zero when disabled.
    pub fn relative_part(&self) -> f32 {
        self.relative
    }

    /// The effective threshold for a strongest local maximum of `strongest`.
    fn effective(&self, strongest: f32) -> f32 {
        self.absolute.max(self.relative * strongest)
    }
}

/// Shi-Tomasi corner detector for f32 images in physical units, e.g. thermal
/// frames in °C. The counterpart of [`DetectorContext`](crate::DetectorContext),
/// owning every buffer it touches.
///
/// The pipeline matches the 8-bit detector: Scharr gradients, the mean 3x3
/// structure tensor, its minimum eigenvalue, 3x3 non-maximum suppression,
/// sorting and spacing. Gradients are computed in f32 and the tensor is
/// summed in f64, so neither quantization nor cancellation limits weak
/// structure. For an image whose values are integers, the result equals the
/// 8-bit detector's on the same values bit for bit, as long as the tensor
/// determinant stays below 2^53, which holds for all but extreme 8-bit
/// contrast. See [`ScoreThreshold`] for the score unit and the threshold.
///
/// Conventions, shared with the 8-bit detector:
/// - Border pixels have zero gradient but count as window samples, and the
///   outermost pixels are never returned.
/// - A pixel is a local maximum when no 8-neighbor scores higher, so equal
///   neighbors on a plateau are all kept.
/// - Results are sorted by descending score, then by row, then by column.
///
/// Buffers grow on demand and are reused. Create the context with
/// [`with_capacity`](Self::with_capacity) to keep allocation out of the
/// per-frame path; calls within the [`DetectorCapacity`] then perform no heap
/// allocation, whatever the image content.
#[derive(Default)]
pub struct FloatDetectorContext {
    planes: Planes<f64>,
    selection: Selection,
}

impl FloatDetectorContext {
    /// Creates an empty context. Buffers grow to fit on the first detection
    /// and are reused thereafter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a context with every buffer allocated for `capacity`.
    pub fn with_capacity(capacity: DetectorCapacity) -> Self {
        let mut context = Self::default();
        context.reserve(capacity);
        context
    }

    /// Allocates every buffer for `capacity` now, so that later calls within
    /// it do not allocate. Existing capacity is kept.
    pub fn reserve(&mut self, capacity: DetectorCapacity) {
        self.planes.reserve(&capacity);
        self.selection.reserve(&capacity);
    }

    /// Detects corners scoring at least `threshold`, at least `min_distance`
    /// pixels apart (zero disables spacing), greedily by descending score.
    ///
    /// Returns `(x, y, score)` from the context's own result buffer; the slice
    /// is valid until the next call. Images narrower or shorter than three
    /// pixels, and flat images, yield no corners.
    pub fn detect(
        &mut self,
        image: FloatImageView<'_>,
        threshold: ScoreThreshold,
        min_distance: u32,
    ) -> &[(u32, u32, f32)] {
        self.detect_candidates(image, threshold);
        let (width, height) = image.dimensions();
        self.selection
            .spaced(&self.planes.candidates, min_distance, width, height)
    }

    /// Detects up to `max_per_cell` corners per cell of a `grid_cols` x
    /// `grid_rows` grid, counting `existing_points` against each cell's budget
    /// and keeping `min_distance` to them, as
    /// [`good_features_to_track_grid`](crate::good_features_to_track_grid) does.
    ///
    /// # Panics
    /// Panics if `grid_cols` or `grid_rows` is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn detect_grid(
        &mut self,
        image: FloatImageView<'_>,
        grid_cols: u32,
        grid_rows: u32,
        max_per_cell: u32,
        threshold: ScoreThreshold,
        min_distance: u32,
        existing_points: &[(f32, f32)],
    ) -> &[(u32, u32, f32)] {
        let grid = GridLayout::new(grid_cols, grid_rows, max_per_cell);
        if max_per_cell == 0 {
            return self.selection.clear();
        }
        self.detect_candidates(image, threshold);
        let (width, height) = image.dimensions();
        self.selection.grid(
            &self.planes.candidates,
            width,
            height,
            grid,
            min_distance,
            existing_points,
        )
    }

    /// Candidates passing `threshold`, sorted, before spacing.
    fn detect_candidates(&mut self, image: FloatImageView<'_>, threshold: ScoreThreshold) {
        self.planes.candidates.clear();
        if image.width() < 3 || image.height() < 3 {
            return;
        }
        run(&mut self.planes, image, threshold);
        let candidates = &mut self.planes.candidates;
        let strongest = candidates.iter().fold(0.0f32, |max, p| max.max(p.2));
        let threshold = threshold.effective(strongest);
        candidates.retain(|&(_, _, score)| score >= threshold);
        sort_by_score(candidates);
    }
}

/// Gradients, structure tensor, scores and non-maximum suppression, leaving
/// the local maxima that can pass `threshold` in row-major order. On x86 an
/// AVX2 copy is selected at runtime.
fn run(planes: &mut Planes<f64>, image: FloatImageView<'_>, threshold: ScoreThreshold) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { run_avx2(planes, image, threshold) };
    }
    run_impl(planes, image, threshold)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn run_avx2(planes: &mut Planes<f64>, image: FloatImageView<'_>, threshold: ScoreThreshold) {
    run_impl(planes, image, threshold)
}

#[inline(always)]
fn run_impl(p: &mut Planes<f64>, image: FloatImageView<'_>, threshold: ScoreThreshold) {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let n = w * h;
    p.grad_x.resize(n, 0.0);
    p.grad_y.resize(n, 0.0);
    p.scores.resize(n, 0.0);

    scharr_f32_into(image.pixels(), w, h, &mut p.grad_x, &mut p.grad_y);
    min_eigenvalue_scores(&p.grad_x, &p.grad_y, w, &mut p.tensor, &mut p.scores);
    // The relative part needs the strongest score, which costs a pass.
    let relative_floor = if threshold.relative > 0.0 {
        quality_floor(&p.scores, w, h, threshold.relative)
    } else {
        0.0
    };
    let floor = threshold.absolute.max(relative_floor);
    local_maxima(
        &p.scores,
        w,
        h,
        floor,
        &mut p.maxima_rows,
        &mut p.candidates,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::TensorSum;

    #[test]
    fn threshold_combines_absolute_and_relative_parts() {
        let absolute = ScoreThreshold::absolute(2.0);
        assert_eq!(absolute.relative_part(), 0.0);
        assert_eq!(absolute.effective(100.0), 2.0);
        let both = absolute.with_relative(0.1);
        assert_eq!(both.effective(100.0), 10.0);
        assert_eq!(both.effective(5.0), 2.0);
        assert_eq!(
            ScoreThreshold::absolute(0.0)
                .with_relative(1.0)
                .effective(3.0),
            3.0
        );
    }

    #[test]
    #[should_panic(expected = "finite and non-negative")]
    fn negative_absolute_threshold_panics() {
        ScoreThreshold::absolute(-1e-9);
    }

    #[test]
    #[should_panic(expected = "finite and non-negative")]
    fn nan_absolute_threshold_panics() {
        ScoreThreshold::absolute(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "within 0..=1")]
    fn relative_threshold_above_one_panics() {
        ScoreThreshold::absolute(0.0).with_relative(1.5);
    }

    #[test]
    #[should_panic(expected = "within 0..=1")]
    fn nan_relative_threshold_panics() {
        ScoreThreshold::absolute(0.0).with_relative(f32::NAN);
    }

    /// Minimum eigenvalue of a window tensor in f64 throughout.
    fn reference(a: f64, b: f64, c: f64) -> f64 {
        let determinant = a * b - c * c;
        let largest = (a + b + ((a - b).powi(2) + 4.0 * c * c).sqrt()) / 2.0;
        if determinant > 0.0 {
            determinant / largest
        } else {
            0.0
        }
    }

    #[test]
    fn f64_tensor_scores_match_reference_for_real_valued_gradients() {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        // Gradient magnitudes from tiny physical slopes to the largest
        // permitted ones, with nearly collinear windows. Below about 1e-6,
        // nearly collinear windows lose precision once the determinant is a
        // subnormal f32; their scores are then negligible anyway.
        for scale in [1e-6f32, 1e-3, 0.37, 41.0, 5e8] {
            for collinear in [false, true] {
                let (dx, dy) = (next(), next());
                let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
                for _ in 0..9 {
                    let jitter = if collinear { 1e-3 } else { 1.0 };
                    let t = next();
                    let gx = (t * dx + next() * jitter) * scale;
                    let gy = (t * dy + next() * jitter) * scale;
                    let (xx, yy, xy) = <f64 as TensorSum>::products(gx, gy);
                    (a, b, c) = (a + xx, b + yy, c + xy);
                }
                let score = <f64 as TensorSum>::scaled_min_eigenvalue(a, b, c, 1.0);
                let expected = reference(a, b, c);
                assert!(
                    (score as f64 - expected).abs() <= expected * 1e-5,
                    "scale {scale}, collinear {collinear}: {score} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn extreme_permitted_values_keep_scores_finite() {
        let max = FloatImageView::MAX_ABS_VALUE;
        let (width, height) = (8u32, 8u32);
        let pixels: Vec<f32> = (0..width * height)
            .map(|i| {
                if (i / width + i % width) % 2 == 0 {
                    max
                } else {
                    -max
                }
            })
            .collect();
        let checker: Vec<f32> = (0..width * height)
            .map(|i| {
                if (i / width < 4) == (i % width < 4) {
                    max
                } else {
                    -max
                }
            })
            .collect();
        for pixels in [pixels, checker] {
            let view = FloatImageView::new(&pixels, width, height).unwrap();
            let mut context = FloatDetectorContext::new();
            context.detect(view, ScoreThreshold::absolute(0.0), 0);
            assert!(
                context
                    .planes
                    .scores
                    .iter()
                    .all(|s| s.is_finite() && *s >= 0.0)
            );
        }
    }
}
