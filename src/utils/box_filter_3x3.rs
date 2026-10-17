#[cfg(test)]
use image::{ImageBuffer, Luma};

/// Centered separable 3x3 average. At boundaries only the available samples
/// contribute; each pass truncates integer division toward zero. Empty,
/// single-row, and single-column images are supported. The library itself
/// uses [`box_filter_3x3_raw`] with a reused scratch buffer.
#[cfg(test)]
pub fn box_filter_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = image.dimensions();
    let mut scratch = Vec::new();
    box_filter_3x3_raw(image, width as usize, height as usize, &mut scratch);
}

/// [`box_filter_3x3_in_place`] over a raw row-major `width * height` buffer.
///
/// `scratch` holds two rows and is resized as needed; passing the same vector
/// across calls makes the filter allocation-free in steady state. Both passes
/// run row by row so the inner loops are contiguous and vectorizable, and the
/// interior divides by the constant 3 while the one-pixel border is handled
/// separately.
#[inline(always)]
pub fn box_filter_3x3_raw(data: &mut [i16], width: usize, height: usize, scratch: &mut Vec<i16>) {
    debug_assert_eq!(data.len(), width * height);
    if width == 0 || height == 0 {
        return;
    }
    scratch.resize(2 * width, 0);
    horizontal_pass(data, width, height, &mut scratch[..width]);
    vertical_pass(data, width, height, scratch);
}

/// Each row becomes the average of itself and its horizontal neighbours.
#[inline(always)]
fn horizontal_pass(data: &mut [i16], width: usize, height: usize, row_copy: &mut [i16]) {
    if width == 1 {
        return;
    }
    for y in 0..height {
        let row = &mut data[y * width..(y + 1) * width];
        row_copy.copy_from_slice(row);
        let src = &*row_copy;

        row[0] = ((src[0] as i32 + src[1] as i32) / 2) as i16;
        let interior = &mut row[1..width - 1];
        for (x, out) in interior.iter_mut().enumerate() {
            let sum = src[x] as i32 + src[x + 1] as i32 + src[x + 2] as i32;
            *out = (sum / 3) as i16;
        }
        row[width - 1] = ((src[width - 2] as i32 + src[width - 1] as i32) / 2) as i16;
    }
}

/// Each row becomes the average of itself and its vertical neighbours, using
/// the original (pre-pass) values of the rows above and below.
#[inline(always)]
fn vertical_pass(data: &mut [i16], width: usize, height: usize, scratch: &mut [i16]) {
    if height == 1 {
        return;
    }
    let (above_copy, current_copy) = scratch.split_at_mut(width);
    for y in 0..height {
        // Rows below `y` are still original; `above_copy` preserves row y-1.
        let (head, tail) = data.split_at_mut((y + 1) * width);
        let row = &mut head[y * width..];
        current_copy.copy_from_slice(row);

        if y == 0 {
            let below = &tail[..width];
            for x in 0..width {
                row[x] = ((current_copy[x] as i32 + below[x] as i32) / 2) as i16;
            }
        } else if y + 1 < height {
            let below = &tail[..width];
            for x in 0..width {
                let sum = above_copy[x] as i32 + current_copy[x] as i32 + below[x] as i32;
                row[x] = (sum / 3) as i16;
            }
        } else {
            for x in 0..width {
                row[x] = ((above_copy[x] as i32 + current_copy[x] as i32) / 2) as i16;
            }
        }
        above_copy.copy_from_slice(current_copy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(source: &ImageBuffer<Luma<i16>, Vec<i16>>) -> ImageBuffer<Luma<i16>, Vec<i16>> {
        let (width, height) = source.dimensions();
        let mut horizontal = source.clone();
        let mut output = source.clone();
        for y in 0..height {
            for x in 0..width {
                let lo = x.saturating_sub(1);
                let hi = (x + 1).min(width - 1);
                let sum: i32 = (lo..=hi).map(|sx| source.get_pixel(sx, y)[0] as i32).sum();
                horizontal.put_pixel(x, y, Luma([(sum / (hi - lo + 1) as i32) as i16]));
            }
        }
        for y in 0..height {
            for x in 0..width {
                let lo = y.saturating_sub(1);
                let hi = (y + 1).min(height - 1);
                let sum: i32 = (lo..=hi)
                    .map(|sy| horizontal.get_pixel(x, sy)[0] as i32)
                    .sum();
                output.put_pixel(x, y, Luma([(sum / (hi - lo + 1) as i32) as i16]));
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
                        0 => i16::MIN,
                        1 => i16::MAX,
                        2 => -17,
                        _ => 29,
                    };
                    Luma([value])
                });
                let expected = reference(&image);
                box_filter_3x3_in_place(&mut image);
                assert_eq!(image, expected, "box filter mismatch at {width}x{height}");
            }
        }
    }

    #[test]
    fn centered_impulse_has_symmetric_support_without_scan_direction_bias() {
        let mut image = ImageBuffer::new(7, 7);
        image.put_pixel(3, 3, Luma([90]));
        box_filter_3x3_in_place(&mut image);
        for y in 0..7 {
            for x in 0..7 {
                let expected = if (2..=4).contains(&x) && (2..=4).contains(&y) {
                    10
                } else {
                    0
                };
                assert_eq!(image.get_pixel(x, y)[0], expected, "at ({x}, {y})");
            }
        }
    }
}
