use std::fmt;

/// A borrowed, validated, row-major f32 image, e.g. a thermal frame in °C.
///
/// The f32 detector and tracker work in the image's own unit: no range
/// normalization and no 8-bit conversion takes place. Validation guarantees
/// that every value is finite and at most [`MAX_ABS_VALUE`](Self::MAX_ABS_VALUE)
/// in magnitude, so no intermediate can overflow and no NaN can reach the
/// score comparisons. Replace invalid pixels (e.g. dead sensor pixels) before
/// creating the view.
#[derive(Clone, Copy, Debug)]
pub struct FloatImageView<'a> {
    pixels: &'a [f32],
    width: u32,
    height: u32,
}

impl<'a> FloatImageView<'a> {
    /// Largest accepted pixel magnitude, 2^24. Within it, Scharr gradients,
    /// structure-tensor sums and their determinant stay finite.
    pub const MAX_ABS_VALUE: f32 = 16_777_216.0;

    /// Validates `pixels` as a `width` x `height` row-major image.
    ///
    /// Costs one pass over the pixels.
    pub fn new(pixels: &'a [f32], width: u32, height: u32) -> Result<Self, FloatImageError> {
        let expected = width as u64 * height as u64;
        if pixels.len() as u64 != expected {
            return Err(FloatImageError::SizeMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        if let Some(index) = first_invalid(pixels) {
            return Err(FloatImageError::InvalidValue {
                index,
                value: pixels[index],
            });
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// `(width, height)` in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The validated pixels, row by row.
    pub fn pixels(&self) -> &'a [f32] {
        self.pixels
    }
}

/// Why [`FloatImageView::new`] rejected an image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FloatImageError {
    /// The pixel slice does not hold exactly `width * height` values.
    SizeMismatch {
        /// `width * height`.
        expected: u64,
        /// Length of the pixel slice.
        actual: usize,
    },
    /// A pixel is NaN, infinite, or larger in magnitude than
    /// [`FloatImageView::MAX_ABS_VALUE`]. Reports the first such pixel.
    InvalidValue {
        /// Row-major index of the pixel.
        index: usize,
        /// Its value.
        value: f32,
    },
}

impl fmt::Display for FloatImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeMismatch { expected, actual } => {
                write!(f, "image needs {expected} pixels, got {actual}")
            }
            Self::InvalidValue { index, value } => write!(
                f,
                "pixel {index} is {value}; values must be finite and at most {} in magnitude",
                FloatImageView::MAX_ABS_VALUE
            ),
        }
    }
}

impl std::error::Error for FloatImageError {}

/// Index of the first pixel that is NaN, infinite or out of range.
fn first_invalid(pixels: &[f32]) -> Option<usize> {
    // NaN fails the comparison, so it is invalid as well.
    let valid = |value: f32| value.abs() <= FloatImageView::MAX_ABS_VALUE;
    const CHUNK: usize = 256;
    for (chunk_index, chunk) in pixels.chunks(CHUNK).enumerate() {
        // A branch-free reduction per chunk keeps the common case vectorized.
        let all_valid = chunk.iter().fold(true, |all, &value| all & valid(value));
        if !all_valid {
            let offset = chunk.iter().position(|&value| !valid(value));
            return offset.map(|offset| chunk_index * CHUNK + offset);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_finite_values_within_range_and_empty_images() {
        let max = FloatImageView::MAX_ABS_VALUE;
        let pixels = [0.0, -0.0, 1e-30, -max, max, 25.5];
        let view = FloatImageView::new(&pixels, 3, 2).unwrap();
        assert_eq!(view.dimensions(), (3, 2));
        assert_eq!(view.pixels(), &pixels);
        assert!(FloatImageView::new(&[], 0, 7).is_ok());
    }

    #[test]
    fn rejects_size_mismatch_without_overflow() {
        assert_eq!(
            FloatImageView::new(&[0.0; 5], 2, 3).unwrap_err(),
            FloatImageError::SizeMismatch {
                expected: 6,
                actual: 5
            }
        );
        let error = FloatImageView::new(&[0.0; 4], u32::MAX, u32::MAX).unwrap_err();
        assert!(matches!(error, FloatImageError::SizeMismatch { .. }));
    }

    #[test]
    fn reports_the_first_invalid_value() {
        let max = FloatImageView::MAX_ABS_VALUE;
        for (bad, index) in [
            (f32::NAN, 0),
            (f32::INFINITY, 300),
            (f32::NEG_INFINITY, 511),
            (max * 2.0, 999),
            (-max * 2.0, 257),
        ] {
            let mut pixels = vec![1.0f32; 1000];
            pixels[index] = bad;
            // A later invalid value must not hide the first one.
            if index + 2 < pixels.len() {
                pixels[index + 2] = f32::NAN;
            }
            match FloatImageView::new(&pixels, 100, 10).unwrap_err() {
                FloatImageError::InvalidValue { index: got, value } => {
                    assert_eq!(got, index);
                    assert_eq!(value.to_bits(), bad.to_bits());
                }
                other => panic!("unexpected {other:?}"),
            }
        }
    }
}
