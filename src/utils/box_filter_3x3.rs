#[cfg(test)]
use image::{ImageBuffer, Luma};

/// Largest input magnitude for which a full 3x3 sum cannot overflow i32.
const MAX_BOX_SUM_INPUT: i32 = i32::MAX / 9;

use std::ops::Add;

/// Horizontal pass of a centered 3x3 box sum over one row, without dividing:
/// `dst[x] = src[x - 1] + src[x] + src[x + 1]`. Boundaries include only the
/// available samples; the caller normalizes by their count when needed. Inputs
/// must have magnitude <= i32::MAX / 9 so both passes fit in i32.
#[inline(always)]
pub fn sum_row_3(src: &[i32], dst: &mut [i32]) {
    // Release builds would wrap silently on overflow; catch violations in debug.
    debug_assert!(
        src.iter()
            .all(|&v| v.unsigned_abs() <= MAX_BOX_SUM_INPUT.unsigned_abs()),
        "box sum input exceeds i32::MAX / 9"
    );
    centered_row_sum(src, dst);
}

/// Vertical pass of a centered 3x3 box sum: lane-wise sum of the horizontally
/// summed rows above, at and below the output row. `None` marks a row outside
/// the image, so boundaries again include only the available samples.
#[inline(always)]
pub fn sum_rows_3(above: Option<&[i32]>, center: &[i32], below: Option<&[i32]>, dst: &mut [i32]) {
    vertical_sum(above, center, below, dst);
}

/// [`sum_row_3`] for f64 rows, which cannot overflow for finite f32 products.
#[inline(always)]
pub fn sum_row_3_f64(src: &[f64], dst: &mut [f64]) {
    centered_row_sum(src, dst);
}

/// [`sum_rows_3`] for f64 rows.
#[inline(always)]
pub fn sum_rows_3_f64(
    above: Option<&[f64]>,
    center: &[f64],
    below: Option<&[f64]>,
    dst: &mut [f64],
) {
    vertical_sum(above, center, below, dst);
}

#[inline(always)]
fn centered_row_sum<T: Copy + Add<Output = T>>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    let width = src.len();
    match width {
        0 => return,
        1 => {
            dst[0] = src[0];
            return;
        }
        _ => {}
    }
    dst[0] = src[0] + src[1];
    dst[width - 1] = src[width - 2] + src[width - 1];
    // Reading from a separate source row keeps this a centered sum rather than
    // a recursive filter, and leaves no loop-carried dependency to vectorize.
    let interior = dst[1..width - 1]
        .iter_mut()
        .zip(&src[..width - 2])
        .zip(&src[1..width - 1])
        .zip(&src[2..]);
    for (((value, &left), &center), &right) in interior {
        *value = left + center + right;
    }
}

#[inline(always)]
fn vertical_sum<T: Copy + Add<Output = T>>(
    above: Option<&[T]>,
    center: &[T],
    below: Option<&[T]>,
    dst: &mut [T],
) {
    debug_assert_eq!(center.len(), dst.len());
    match (above, below) {
        (Some(above), Some(below)) => {
            let rows = dst.iter_mut().zip(above).zip(center).zip(below);
            for (((value, &above), &center), &below) in rows {
                *value = above + center + below;
            }
        }
        (Some(edge), None) | (None, Some(edge)) => {
            for ((value, &edge), &center) in dst.iter_mut().zip(edge).zip(center) {
                *value = edge + center;
            }
        }
        (None, None) => dst.copy_from_slice(center),
    }
}

/// Whole-image composition of the row kernels, so tests can check them against
/// an out-of-place 2D reference. Empty, single-row, and single-column images
/// are supported.
#[cfg(test)]
pub fn box_sum_3x3_in_place(image: &mut ImageBuffer<Luma<i32>, Vec<i32>>) {
    let width = image.width() as usize;
    if width == 0 || image.height() == 0 {
        return;
    }
    let data: &mut [i32] = image;
    let mut horizontal = vec![0i32; data.len()];
    for (src, dst) in data
        .chunks_exact(width)
        .zip(horizontal.chunks_exact_mut(width))
    {
        sum_row_3(src, dst);
    }
    let rows: Vec<&[i32]> = horizontal.chunks_exact(width).collect();
    for (y, dst) in data.chunks_exact_mut(width).enumerate() {
        let above = y.checked_sub(1).map(|above| rows[above]);
        sum_rows_3(above, rows[y], rows.get(y + 1).copied(), dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(source: &ImageBuffer<Luma<i32>, Vec<i32>>) -> ImageBuffer<Luma<i32>, Vec<i32>> {
        let (width, height) = source.dimensions();
        let mut horizontal = source.clone();
        let mut output = source.clone();
        for y in 0..height {
            for x in 0..width {
                let lo = x.saturating_sub(1);
                let hi = (x + 1).min(width - 1);
                let sum: i64 = (lo..=hi).map(|sx| source.get_pixel(sx, y)[0] as i64).sum();
                horizontal.put_pixel(x, y, Luma([i32::try_from(sum).unwrap()]));
            }
        }
        for y in 0..height {
            for x in 0..width {
                let lo = y.saturating_sub(1);
                let hi = (y + 1).min(height - 1);
                let sum: i64 = (lo..=hi)
                    .map(|sy| horizontal.get_pixel(x, sy)[0] as i64)
                    .sum();
                output.put_pixel(x, y, Luma([i32::try_from(sum).unwrap()]));
            }
        }
        output
    }

    #[test]
    fn matches_out_of_place_reference_including_small_dimensions() {
        for width in 0..9 {
            for height in 0..9 {
                let mut image = ImageBuffer::from_fn(width, height, |x, y| {
                    let value = match (x + 3 * y) % 4 {
                        0 => -16_646_400,
                        1 => 16_646_400,
                        2 => -17,
                        _ => 29,
                    };
                    Luma([value])
                });
                let expected = reference(&image);
                box_sum_3x3_in_place(&mut image);
                assert_eq!(image, expected, "box sum mismatch at {width}x{height}");
            }
        }
    }

    #[test]
    fn centered_impulse_has_symmetric_support_without_scan_direction_bias() {
        let mut image = ImageBuffer::new(7, 7);
        image.put_pixel(3, 3, Luma([90]));
        box_sum_3x3_in_place(&mut image);
        for y in 0..7 {
            for x in 0..7 {
                let expected = if (2..=4).contains(&x) && (2..=4).contains(&y) {
                    90
                } else {
                    0
                };
                assert_eq!(image.get_pixel(x, y)[0], expected, "at ({x}, {y})");
            }
        }
    }

    #[test]
    fn maximum_scharr_products_fit_at_interior_and_boundaries() {
        let product = 4080 * 4080;
        let mut image = ImageBuffer::from_pixel(3, 3, Luma([product]));
        box_sum_3x3_in_place(&mut image);
        assert_eq!(image.get_pixel(1, 1)[0], 149_817_600);
        assert_eq!(image.get_pixel(0, 0)[0], 4 * product);
        assert_eq!(image.get_pixel(1, 0)[0], 6 * product);
    }

    #[test]
    fn largest_permitted_inputs_do_not_overflow() {
        for value in [MAX_BOX_SUM_INPUT, -MAX_BOX_SUM_INPUT] {
            let mut image = ImageBuffer::from_pixel(3, 3, Luma([value]));
            box_sum_3x3_in_place(&mut image);
            assert_eq!(image.get_pixel(1, 1)[0], 9 * value);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "exceeds i32::MAX / 9")]
    fn out_of_range_input_is_rejected_in_debug_builds() {
        let mut image = ImageBuffer::from_pixel(3, 3, Luma([MAX_BOX_SUM_INPUT + 1]));
        box_sum_3x3_in_place(&mut image);
    }
}
