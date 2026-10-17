use image::{ImageBuffer, Luma};

/// Largest input magnitude for which a full 3x3 sum cannot overflow i32.
const MAX_BOX_SUM_INPUT: i32 = i32::MAX / 9;

/// Centered separable 3x3 sum without allocating or dividing. Boundaries include
/// only available samples; the caller normalizes by their count when needed.
/// Inputs must have magnitude <= i32::MAX / 9 so both passes fit in i32.
/// Empty, single-row, and single-column images are supported.
pub fn box_sum_3x3_in_place(image: &mut ImageBuffer<Luma<i32>, Vec<i32>>) {
    if image.width() == 0 || image.height() == 0 {
        return;
    }
    // Release builds would wrap silently on overflow; catch violations in debug.
    debug_assert!(
        image
            .iter()
            .all(|&v| v.unsigned_abs() <= MAX_BOX_SUM_INPUT.unsigned_abs()),
        "box_sum_3x3_in_place input exceeds i32::MAX / 9"
    );
    // Accumulate each dimension once; normalization is deferred to the score.
    box_sum_horizontal_3x3_in_place(image);
    box_sum_vertical_3x3_in_place(image);
}

fn box_sum_horizontal_3x3_in_place(image: &mut ImageBuffer<Luma<i32>, Vec<i32>>) {
    let width = image.width() as usize;
    let data: &mut [i32] = image;

    // Read from a copy of the row: rereading a written pixel would make this a
    // recursive filter, and a register chain would serialize the loop.
    let mut source = vec![0i32; width];
    for row in data.chunks_exact_mut(width) {
        source.copy_from_slice(row);
        sum_row_3(&source, row);
    }
}

/// `dst[x] = src[x - 1] + src[x] + src[x + 1]`, omitting neighbors outside the
/// row. The interior has no loop-carried dependency, so it vectorizes.
fn sum_row_3(src: &[i32], dst: &mut [i32]) {
    let width = src.len();
    if width == 1 {
        dst[0] = src[0];
        return;
    }
    dst[0] = src[0] + src[1];
    dst[width - 1] = src[width - 2] + src[width - 1];
    let interior = dst[1..width - 1]
        .iter_mut()
        .zip(&src[..width - 2])
        .zip(&src[1..width - 1])
        .zip(&src[2..]);
    for (((value, &left), &center), &right) in interior {
        *value = left + center + right;
    }
}

fn box_sum_vertical_3x3_in_place(image: &mut ImageBuffer<Luma<i32>, Vec<i32>>) {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let data: &mut [i32] = image;

    // Sum whole rows lane by lane so the pass streams through memory and
    // vectorizes; a column-wise walk strides a full row per element. `above`
    // keeps the original values of the previous row, which is overwritten.
    let mut above = vec![0i32; width];
    for y in 0..height {
        let (head, tail) = data.split_at_mut((y + 1) * width);
        let row = &mut head[y * width..];
        match tail.get(..width) {
            Some(below) => {
                for ((value, above), &below) in row.iter_mut().zip(&mut above).zip(below) {
                    let center = *value;
                    *value = *above + center + below;
                    *above = center;
                }
            }
            None => {
                for (value, above) in row.iter_mut().zip(&mut above) {
                    let center = *value;
                    *value = *above + center;
                    *above = center;
                }
            }
        }
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
