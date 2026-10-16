use image::{ImageBuffer, Luma};

/// Centered separable 3x3 average without allocating. At boundaries only the
/// available samples contribute; each pass truncates integer division toward
/// zero. Empty, single-row, and single-column images are supported.
pub fn box_filter_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    if image.width() == 0 || image.height() == 0 {
        return;
    }
    // Apply the separable 3x3 box filter: first across rows, then across columns
    box_filter_horizontal_3x3_in_place(image);
    box_filter_vertical_3x3_in_place(image);
}

fn box_filter_horizontal_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let data: &mut [i16] = image;

    for y in 0..height {
        let row = y * width;

        // Keep original neighbors in registers: rereading a written pixel would
        // make this a recursive filter rather than a centered box average.
        let mut left = 0i32;
        let mut center = data[row] as i32;
        for x in 0..width {
            let right = if x + 1 < width {
                data[row + x + 1] as i32
            } else {
                0
            };
            let count = 1 + i32::from(x > 0) + i32::from(x + 1 < width);
            data[row + x] = ((left + center + right) / count) as i16;
            left = center;
            center = right;
        }
    }
}

fn box_filter_vertical_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let data: &mut [i16] = image;

    for x in 0..width {
        let mut above = 0i32;
        let mut center = data[x] as i32;
        for y in 0..height {
            let below = if y + 1 < height {
                data[(y + 1) * width + x] as i32
            } else {
                0
            };
            let count = 1 + i32::from(y > 0) + i32::from(y + 1 < height);
            data[y * width + x] = ((above + center + below) / count) as i16;
            above = center;
            center = below;
        }
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
