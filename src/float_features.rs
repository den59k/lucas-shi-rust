use crate::features::{Corner, Detector, DetectorCapacity, DetectorImage, Grid, Threshold};
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
    #[must_use]
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
    #[must_use]
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

    fn threshold(self) -> Threshold {
        Threshold {
            absolute: self.absolute,
            relative: self.relative,
        }
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
    detector: Detector<f64>,
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
        self.detector.reserve(&capacity);
    }

    /// Detects corners scoring at least `threshold`, at least `min_distance`
    /// pixels apart (zero disables spacing), greedily by descending score.
    ///
    /// Returns [`Corner`]s from the context's own result buffer; the slice is
    /// valid until the next call. Images narrower or shorter than three
    /// pixels, and flat images, yield no corners.
    pub fn detect(
        &mut self,
        image: FloatImageView<'_>,
        threshold: ScoreThreshold,
        min_distance: u32,
    ) -> &[Corner] {
        self.detector
            .detect(image, threshold.threshold(), min_distance)
    }

    /// Detects up to `grid.max_per_cell` corners per cell of `grid`, counting
    /// `existing_points` against each cell's budget and keeping `min_distance`
    /// to them, as
    /// [`good_features_to_track_grid`](crate::good_features_to_track_grid) does.
    ///
    /// # Panics
    /// Panics if `grid.cols` or `grid.rows` is zero.
    pub fn detect_grid(
        &mut self,
        image: FloatImageView<'_>,
        grid: Grid,
        threshold: ScoreThreshold,
        min_distance: u32,
        existing_points: &[(f32, f32)],
    ) -> &[Corner] {
        self.detector.detect_grid(
            image,
            grid,
            threshold.threshold(),
            min_distance,
            existing_points,
        )
    }
}

impl DetectorImage for FloatImageView<'_> {
    type Sum = f64;

    fn dimensions(self) -> (u32, u32) {
        (self.width(), self.height())
    }

    #[inline(always)]
    fn gradients_into(self, grad_x: &mut [f32], grad_y: &mut [f32]) {
        let (width, height) = (self.width() as usize, self.height() as usize);
        scharr_f32_into(self.pixels(), width, height, grad_x, grad_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_passes_both_parts_to_the_detector() {
        let absolute = ScoreThreshold::absolute(2.0);
        assert_eq!(absolute.relative_part(), 0.0);
        let both = absolute.with_relative(0.1);
        assert_eq!((both.absolute_part(), both.relative_part()), (2.0, 0.1));
        assert_eq!(
            both.threshold(),
            Threshold {
                absolute: 2.0,
                relative: 0.1
            }
        );
    }

    #[test]
    #[should_panic(expected = "finite and non-negative")]
    fn negative_absolute_threshold_panics() {
        let _ = ScoreThreshold::absolute(-1e-9);
    }

    #[test]
    #[should_panic(expected = "finite and non-negative")]
    fn nan_absolute_threshold_panics() {
        let _ = ScoreThreshold::absolute(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "within 0..=1")]
    fn relative_threshold_above_one_panics() {
        let _ = ScoreThreshold::absolute(0.0).with_relative(1.5);
    }

    #[test]
    #[should_panic(expected = "within 0..=1")]
    fn nan_relative_threshold_panics() {
        let _ = ScoreThreshold::absolute(0.0).with_relative(f32::NAN);
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
                    .detector
                    .scores()
                    .iter()
                    .all(|s| s.is_finite() && *s >= 0.0)
            );
        }
    }
}
