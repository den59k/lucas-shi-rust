/// Scharr gradients of a row-major f32 image into caller-provided buffers
/// (length `width * height` each), performing no heap allocation.
///
/// Like the 8-bit gradients, the outputs are 32 times the slope per pixel and
/// zero on the image border. Differences of opposite pixels are formed before
/// weighting, so integer-valued images give exactly the 8-bit gradients and a
/// constant offset cancels whenever the differences are exact. The loops are
/// written for the compiler to vectorize.
#[inline(always)]
pub fn scharr_f32_into(
    pixels: &[f32],
    width: usize,
    height: usize,
    grad_x: &mut [f32],
    grad_y: &mut [f32],
) {
    let pixel_count = width * height;
    assert!(pixels.len() == pixel_count && grad_x.len() == pixel_count);
    assert_eq!(grad_y.len(), pixel_count);
    if width < 3 || height < 3 {
        grad_x.fill(0.0);
        grad_y.fill(0.0);
        return;
    }
    let last_row = (height - 1) * width;
    for gradient in [&mut *grad_x, &mut *grad_y] {
        gradient[..width].fill(0.0);
        gradient[last_row..].fill(0.0);
    }

    let row = |y: usize| y * width..(y + 1) * width;
    for y in 1..height - 1 {
        let (gx, gy) = (&mut grad_x[row(y)], &mut grad_y[row(y)]);
        for gradient in [&mut *gx, &mut *gy] {
            gradient[0] = 0.0;
            gradient[width - 1] = 0.0;
        }
        let (up, middle, down) = (&pixels[row(y - 1)], &pixels[row(y)], &pixels[row(y + 1)]);
        scharr_row(
            up,
            middle,
            down,
            &mut gx[1..width - 1],
            &mut gy[1..width - 1],
        );
    }
}

/// [`scharr_f32_into`], compiled for AVX2 when the CPU supports it. For
/// callers that are not already inside an AVX2 function.
pub fn scharr_f32(
    pixels: &[f32],
    width: usize,
    height: usize,
    grad_x: &mut [f32],
    grad_y: &mut [f32],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { scharr_f32_avx2(pixels, width, height, grad_x, grad_y) };
    }
    scharr_f32_into(pixels, width, height, grad_x, grad_y)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn scharr_f32_avx2(
    pixels: &[f32],
    width: usize,
    height: usize,
    grad_x: &mut [f32],
    grad_y: &mut [f32],
) {
    scharr_f32_into(pixels, width, height, grad_x, grad_y)
}

/// Gradients of one row's interior pixels from the rows above, at and below.
#[inline(always)]
fn scharr_row(up: &[f32], middle: &[f32], down: &[f32], gx: &mut [f32], gy: &mut [f32]) {
    let n = gx.len();
    let (up, middle, down, gy) = (&up[..n + 2], &middle[..n + 2], &down[..n + 2], &mut gy[..n]);
    for i in 0..n {
        let horizontal = |row: &[f32]| row[i + 2] - row[i];
        let vertical = |k: usize| down[i + k] - up[i + k];
        gx[i] = 3.0 * (horizontal(up) + horizontal(down)) + 10.0 * horizontal(middle);
        gy[i] = 3.0 * (vertical(0) + vertical(2)) + 10.0 * vertical(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::fast_gradients::compute_gradients_into;
    use image::{GrayImage, Luma};

    #[test]
    fn integer_images_give_the_8_bit_gradients_exactly() {
        for (width, height) in [(1, 1), (2, 5), (3, 3), (4, 7), (17, 9), (64, 33), (127, 5)] {
            let image = GrayImage::from_fn(width, height, |x, y| {
                Luma([((x * 73 + y * 151) ^ (x * y * 7)) as u8])
            });
            let n = (width * height) as usize;
            let (mut ix, mut iy) = (vec![0i16; n], vec![0i16; n]);
            compute_gradients_into(&image, &mut ix, &mut iy);
            let pixels: Vec<f32> = image.as_raw().iter().map(|&v| v as f32).collect();
            let (mut fx, mut fy) = (vec![f32::NAN; n], vec![f32::NAN; n]);
            scharr_f32_into(&pixels, width as usize, height as usize, &mut fx, &mut fy);
            let expected = |g: &[i16]| g.iter().map(|&v| v as f32).collect::<Vec<_>>();
            assert_eq!(fx, expected(&ix), "gx at {width}x{height}");
            assert_eq!(fy, expected(&iy), "gy at {width}x{height}");
        }
    }

    #[test]
    fn exact_offset_cancels() {
        let (width, height) = (9, 6);
        let base: Vec<f32> = (0..width * height)
            .map(|i| ((i * 37) % 23) as f32 * 0.125)
            .collect();
        let shifted: Vec<f32> = base.iter().map(|&v| v + 300.0).collect();
        let gradients = |pixels: &[f32]| {
            let (mut gx, mut gy) = (vec![0.0; width * height], vec![0.0; width * height]);
            scharr_f32_into(pixels, width, height, &mut gx, &mut gy);
            (gx, gy)
        };
        assert_eq!(gradients(&base), gradients(&shifted));
    }
}
