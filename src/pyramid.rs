use image::{GrayImage, ImageBuffer};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

/// Builds a pyramid of images where each successive layer is half as large in width and height
///
/// This method just takes the average of the 4 pixels, no interpolation or anything like that
///
/// # Arguments
/// * `image` - Source image (grayscale)
/// * `levels` - Level count
///
/// # Returns
/// Vector of layers in descending order of size. First element is source image
pub fn build_pyramid(image: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyramid = Vec::new();
    build_pyramid_into(image, levels, &mut pyramid);
    pyramid
}

/// Builds the pyramid into an existing buffer, reusing each level's storage when
/// its dimensions are unchanged.
///
/// In steady state (same image size and `levels` every call) this performs no
/// heap allocation: each level's pixel buffer is overwritten in place. The
/// `pyramid` is resized to the actual number of produced levels.
pub fn build_pyramid_into(image: &GrayImage, levels: usize, pyramid: &mut Vec<GrayImage>) {
    // Level 0 is a copy of the source into the (reused) buffer.
    ensure_level(pyramid, 0, image.width(), image.height());
    pyramid[0].copy_from_slice(image.as_raw());

    let mut produced = 1;
    for level in 1..levels {
        let (prev_w, prev_h) = pyramid[level - 1].dimensions();

        // Stop when the previous level can no longer be halved.
        if prev_w < 2 || prev_h < 2 {
            break;
        }

        let (new_w, new_h) = (prev_w / 2, prev_h / 2);
        ensure_level(pyramid, level, new_w, new_h);

        // Borrow the previous (read) and current (write) levels disjointly.
        let (head, tail) = pyramid.split_at_mut(level);
        let previous_level = &head[level - 1];
        let new_image = &mut tail[0];

        downsample_2x2_into(
            previous_level.as_raw(),
            prev_w as usize,
            new_w as usize,
            new_h as usize,
            &mut **new_image,
        );

        produced += 1;
    }

    pyramid.truncate(produced);
}

/// Downsamples `src` (`prev_w` wide) by averaging each 2x2 block into one output
/// pixel, writing `new_w * new_h` bytes into `dst`. `prev_w >= 2 * new_w` and the
/// source must have at least `2 * new_h` rows, which the caller guarantees.
///
/// On `wasm32 + simd128` this uses a vectorized path; everywhere else it is a
/// flat-slice scalar loop (no per-pixel bounds-checked `get_pixel`/`put_pixel`).
fn downsample_2x2_into(src: &[u8], prev_w: usize, new_w: usize, new_h: usize, dst: &mut [u8]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { downsample_2x2_simd128(src, prev_w, new_w, new_h, dst) }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        downsample_2x2_scalar(src, prev_w, new_w, new_h, dst);
    }
}

/// Flat-slice scalar 2x2 box downsample. Indices are provably in range
/// (`2*x+1 < prev_w`, `2*y+1 < 2*new_h <= src height`), so reads use unchecked
/// access to keep the loop branch-free on targets without SIMD.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn downsample_2x2_scalar(src: &[u8], prev_w: usize, new_w: usize, new_h: usize, dst: &mut [u8]) {
    for y in 0..new_h {
        let r0 = (2 * y) * prev_w;
        let r1 = r0 + prev_w;
        let out_row = y * new_w;
        for x in 0..new_w {
            let px = 2 * x;
            // SAFETY: r1 + px + 1 = (2y+1)*prev_w + 2x+1 < src.len(), and
            // out_row + x < dst.len(), guaranteed by the caller's dimensions.
            unsafe {
                let s = *src.get_unchecked(r0 + px) as u32
                    + *src.get_unchecked(r0 + px + 1) as u32
                    + *src.get_unchecked(r1 + px) as u32
                    + *src.get_unchecked(r1 + px + 1) as u32;
                *dst.get_unchecked_mut(out_row + x) = (s / 4) as u8;
            }
        }
    }
}

/// WASM `simd128` 2x2 box downsample. Produces 16 output pixels per step:
/// `u16x8_extadd_pairwise_u8x16` does the horizontal 2->1 sum of adjacent source
/// bytes, the two source rows are added, the 2x2 sum (<= 1020) is shifted right
/// by 2 (== integer /4), and the two halves are narrowed back to `u8x16`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn downsample_2x2_simd128(
    src: &[u8],
    prev_w: usize,
    new_w: usize,
    new_h: usize,
    dst: &mut [u8],
) {
    let chunks = new_w / 16;
    for y in 0..new_h {
        let r0 = (2 * y) * prev_w;
        let r1 = r0 + prev_w;
        let out_row = y * new_w;

        for c in 0..chunks {
            let out_x = c * 16;
            let in_x = out_x * 2; // 32 source columns -> 16 outputs
            let (s0a, s0b, s1a, s1b) = unsafe {
                (
                    v128_load(src.as_ptr().add(r0 + in_x) as *const v128),
                    v128_load(src.as_ptr().add(r0 + in_x + 16) as *const v128),
                    v128_load(src.as_ptr().add(r1 + in_x) as *const v128),
                    v128_load(src.as_ptr().add(r1 + in_x + 16) as *const v128),
                )
            };

            // Horizontal adjacent-pair sums (u8 -> u16), then add the two rows.
            let lo = u16x8_add(
                u16x8_extadd_pairwise_u8x16(s0a),
                u16x8_extadd_pairwise_u8x16(s1a),
            );
            let hi = u16x8_add(
                u16x8_extadd_pairwise_u8x16(s0b),
                u16x8_extadd_pairwise_u8x16(s1b),
            );

            // /4 and narrow (values <= 255 after the shift, so no real saturation).
            let packed = u8x16_narrow_i16x8(u16x8_shr(lo, 2), u16x8_shr(hi, 2));
            unsafe {
                v128_store(dst.as_mut_ptr().add(out_row + out_x) as *mut v128, packed);
            }
        }

        // Scalar tail for the columns past the last full chunk of 16.
        for x in (chunks * 16)..new_w {
            let px = 2 * x;
            unsafe {
                let s = *src.get_unchecked(r0 + px) as u32
                    + *src.get_unchecked(r0 + px + 1) as u32
                    + *src.get_unchecked(r1 + px) as u32
                    + *src.get_unchecked(r1 + px + 1) as u32;
                *dst.get_unchecked_mut(out_row + x) = (s / 4) as u8;
            }
        }
    }
}

/// Ensures `pyramid[index]` exists with the given dimensions, allocating only
/// when the slot is missing or its size changed.
fn ensure_level(pyramid: &mut Vec<GrayImage>, index: usize, width: u32, height: u32) {
    if index < pyramid.len() {
        if pyramid[index].dimensions() != (width, height) {
            pyramid[index] = ImageBuffer::new(width, height);
        }
    } else {
        pyramid.push(ImageBuffer::new(width, height));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn make_image(width: u32, height: u32) -> GrayImage {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 37 + y * 19 + (x ^ y) * 7) & 0xff) as u8;
                img.put_pixel(x, y, Luma([v]));
            }
        }
        img
    }

    /// Naive get_pixel reference for one 2x2-averaging level.
    fn reference_half(prev: &GrayImage) -> GrayImage {
        let (pw, ph) = prev.dimensions();
        let (nw, nh) = (pw / 2, ph / 2);
        let mut out = GrayImage::new(nw, nh);
        for y in 0..nh {
            for x in 0..nw {
                let (px, py) = (2 * x, 2 * y);
                let s = prev.get_pixel(px, py)[0] as u32
                    + prev.get_pixel(px + 1, py)[0] as u32
                    + prev.get_pixel(px, py + 1)[0] as u32
                    + prev.get_pixel(px + 1, py + 1)[0] as u32;
                out.put_pixel(x, y, Luma([(s / 4) as u8]));
            }
        }
        out
    }

    #[test]
    fn pyramid_matches_naive_reference() {
        // Odd and even widths/heights exercise the SIMD chunk + scalar tail.
        for (w, h) in [(64u32, 48u32), (65, 49), (37, 71), (16, 16), (3, 3)] {
            let img = make_image(w, h);
            let pyr = build_pyramid(&img, 4);

            assert_eq!(pyr[0], img);
            let mut expected = img.clone();
            for level in pyr.iter().skip(1) {
                expected = reference_half(&expected);
                assert_eq!(level, &expected, "level mismatch at {w}x{h}");
            }
        }
    }
}
