use image::{GrayImage, ImageBuffer, Luma};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Used by every non-SIMD path; unused on wasm32 built with +simd128.
#[allow(dead_code)]
const HORIZONTAL_SCHARR_3X3_OLD: [i32; 9] = [-3, 0, 3, -10, 0, 10, -3, 0, 3];
#[allow(dead_code)]
const VERTICAL_SCHARR_3X3_OLD: [i32; 9] = [-3, -10, -3, 0, 0, 0, 3, 10, 3];

type GradientProduct = (
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
);

/// Computes signed Scharr gradients, allocating fresh output buffers.
///
/// Prefer [`compute_gradients_into`] on a hot path so the gradient buffers can
/// be reused across frames.
pub fn compute_gradients(img: &GrayImage) -> GradientProduct {
    let (width, height) = img.dimensions();
    let mut grad_x = vec![0i16; (width * height) as usize];
    let mut grad_y = vec![0i16; (width * height) as usize];
    compute_gradients_into(img, &mut grad_x, &mut grad_y);
    (
        ImageBuffer::from_vec(width, height, grad_x).unwrap(),
        ImageBuffer::from_vec(width, height, grad_y).unwrap(),
    )
}

/// Computes signed Scharr gradients into caller-provided buffers (length
/// `width * height` each), performing no heap allocation.
///
/// Selection is done per target:
/// - `aarch64`: NEON
/// - `x86`/`x86_64`: runtime AVX2 detection, otherwise scalar fallback
/// - `wasm32`: simd128 when the target was built with `+simd128`, otherwise
///   scalar fallback (WASM SIMD has no runtime detection, so it is a
///   compile-time choice)
/// - everything else: historical scalar implementation
#[cfg(target_arch = "aarch64")]
pub fn compute_gradients_into(img: &GrayImage, grad_x: &mut [i16], grad_y: &mut [i16]) {
    unsafe { compute_gradients_neon_into(img, grad_x, grad_y) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn compute_gradients_into(img: &GrayImage, grad_x: &mut [i16], grad_y: &mut [i16]) {
    unsafe { compute_gradients_simd128_into(img, grad_x, grad_y) }
}

#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
pub fn compute_gradients_into(img: &GrayImage, grad_x: &mut [i16], grad_y: &mut [i16]) {
    compute_gradients_manual_into(
        img,
        &HORIZONTAL_SCHARR_3X3_OLD,
        &VERTICAL_SCHARR_3X3_OLD,
        grad_x,
        grad_y,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn compute_gradients_into(img: &GrayImage, grad_x: &mut [i16], grad_y: &mut [i16]) {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            compute_gradients_avx2_into(img, grad_x, grad_y);
        }
        return;
    }

    compute_gradients_manual_into(
        img,
        &HORIZONTAL_SCHARR_3X3_OLD,
        &VERTICAL_SCHARR_3X3_OLD,
        grad_x,
        grad_y,
    );
}

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86",
    target_arch = "x86_64"
)))]
pub fn compute_gradients_into(img: &GrayImage, grad_x: &mut [i16], grad_y: &mut [i16]) {
    compute_gradients_manual_into(
        img,
        &HORIZONTAL_SCHARR_3X3_OLD,
        &VERTICAL_SCHARR_3X3_OLD,
        grad_x,
        grad_y,
    );
}

// Scalar reference / fallback. Unused on wasm32 built with +simd128.
#[allow(dead_code)]
fn compute_gradients_manual_into(
    img: &GrayImage,
    kernel_x: &[i32; 9],
    kernel_y: &[i32; 9],
    grad_x: &mut [i16],
    grad_y: &mut [i16],
) {
    let (width, height) = img.dimensions();
    let width = width as usize;

    // Borders are never written below, so clear the whole buffer first; this
    // also wipes any data left over from a previous (reused) frame.
    grad_x.fill(0);
    grad_y.fill(0);

    if width < 3 || (height as usize) < 3 {
        return;
    }

    for y in 1..height - 1 {
        for x in 1..width as u32 - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = img.get_pixel(x + kx - 1, y + ky - 1)[0] as i32;
                    gx += pixel * kernel_x[(ky * 3 + kx) as usize];
                    gy += pixel * kernel_y[(ky * 3 + kx) as usize];
                }
            }

            let idx = y as usize * width + x as usize;
            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn compute_gradients_avx2_into(img: &GrayImage, grad_x: &mut [i16], grad_y: &mut [i16]) {
    let (width_u32, height_u32) = img.dimensions();
    let width = width_u32 as usize;
    let height = height_u32 as usize;

    grad_x.fill(0);
    grad_y.fill(0);

    if width < 3 || height < 3 {
        return;
    }

    let src = img.as_raw();

    let coeff3 = _mm256_set1_epi16(3);
    let coeff10 = _mm256_set1_epi16(10);
    let interior_chunks_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let (top, mid, bottom) = unsafe {
            (
                src.as_ptr().add((y - 1) * width),
                src.as_ptr().add(y * width),
                src.as_ptr().add((y + 1) * width),
            )
        };
        let row = y * width;

        let mut x = 1usize;
        while x < interior_chunks_end {
            let (tl, tc, tr, ml, mr, bl, bc, br) = unsafe {
                (
                    load_u8x16_as_i16(top.add(x - 1)),
                    load_u8x16_as_i16(top.add(x)),
                    load_u8x16_as_i16(top.add(x + 1)),
                    load_u8x16_as_i16(mid.add(x - 1)),
                    load_u8x16_as_i16(mid.add(x + 1)),
                    load_u8x16_as_i16(bottom.add(x - 1)),
                    load_u8x16_as_i16(bottom.add(x)),
                    load_u8x16_as_i16(bottom.add(x + 1)),
                )
            };

            let gx3 = _mm256_sub_epi16(_mm256_add_epi16(tr, br), _mm256_add_epi16(tl, bl));
            let gx10 = _mm256_sub_epi16(mr, ml);
            let gx = _mm256_add_epi16(
                _mm256_mullo_epi16(gx3, coeff3),
                _mm256_mullo_epi16(gx10, coeff10),
            );

            let gy3 = _mm256_sub_epi16(_mm256_add_epi16(bl, br), _mm256_add_epi16(tl, tr));
            let gy10 = _mm256_sub_epi16(bc, tc);
            let gy = _mm256_add_epi16(
                _mm256_mullo_epi16(gy3, coeff3),
                _mm256_mullo_epi16(gy10, coeff10),
            );

            unsafe {
                _mm256_storeu_si256(grad_x.as_mut_ptr().add(row + x) as *mut __m256i, gx);
                _mm256_storeu_si256(grad_y.as_mut_ptr().add(row + x) as *mut __m256i, gy);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * width + x + 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y + 1) * width + x - 1] as i32))
                + 10 * (src[y * width + x + 1] as i32 - src[y * width + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * width + x - 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y - 1) * width + x + 1] as i32))
                + 10 * (src[(y + 1) * width + x] as i32 - src[(y - 1) * width + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn load_u8x16_as_i16(ptr: *const u8) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(ptr as *const __m128i)) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compute_gradients_neon_into(img: &GrayImage, grad_x: &mut [i16], grad_y: &mut [i16]) {
    let (width_u32, height_u32) = img.dimensions();
    let width = width_u32 as usize;
    let height = height_u32 as usize;

    grad_x.fill(0);
    grad_y.fill(0);

    if width < 3 || height < 3 {
        return;
    }

    let src = img.as_raw();

    let coeff3 = vdupq_n_s16(3);
    let coeff10 = vdupq_n_s16(10);
    let interior_chunks_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let (top, mid, bottom) = unsafe {
            (
                src.as_ptr().add((y - 1) * width),
                src.as_ptr().add(y * width),
                src.as_ptr().add((y + 1) * width),
            )
        };
        let row = y * width;

        let mut x = 1usize;
        while x < interior_chunks_end {
            let (tl, tc, tr, ml, mr, bl, bc, br) = unsafe {
                (
                    load_u8x16_as_i16x8x2(top.add(x - 1)),
                    load_u8x16_as_i16x8x2(top.add(x)),
                    load_u8x16_as_i16x8x2(top.add(x + 1)),
                    load_u8x16_as_i16x8x2(mid.add(x - 1)),
                    load_u8x16_as_i16x8x2(mid.add(x + 1)),
                    load_u8x16_as_i16x8x2(bottom.add(x - 1)),
                    load_u8x16_as_i16x8x2(bottom.add(x)),
                    load_u8x16_as_i16x8x2(bottom.add(x + 1)),
                )
            };

            let gx_lo = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(tr.0, br.0), vaddq_s16(tl.0, bl.0)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(mr.0, ml.0), coeff10),
            );
            let gx_hi = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(tr.1, br.1), vaddq_s16(tl.1, bl.1)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(mr.1, ml.1), coeff10),
            );
            let gy_lo = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(bl.0, br.0), vaddq_s16(tl.0, tr.0)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(bc.0, tc.0), coeff10),
            );
            let gy_hi = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(bl.1, br.1), vaddq_s16(tl.1, tr.1)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(bc.1, tc.1), coeff10),
            );

            unsafe {
                vst1q_s16(grad_x.as_mut_ptr().add(row + x), gx_lo);
                vst1q_s16(grad_x.as_mut_ptr().add(row + x + 8), gx_hi);
                vst1q_s16(grad_y.as_mut_ptr().add(row + x), gy_lo);
                vst1q_s16(grad_y.as_mut_ptr().add(row + x + 8), gy_hi);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * width + x + 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y + 1) * width + x - 1] as i32))
                + 10 * (src[y * width + x + 1] as i32 - src[y * width + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * width + x - 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y - 1) * width + x + 1] as i32))
                + 10 * (src[(y + 1) * width + x] as i32 - src[(y - 1) * width + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn load_u8x16_as_i16x8x2(ptr: *const u8) -> (int16x8_t, int16x8_t) {
    let bytes = unsafe { vld1q_u8(ptr) };
    (
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes))),
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bytes))),
    )
}

/// WASM `simd128` Scharr gradients. Mirrors the NEON path: 16 source bytes per
/// step are widened into two `i16x8` lanes (low/high), the signed Scharr
/// response is accumulated in 16-bit lanes (it fits: |g| <= 4080), and the two
/// halves are stored back. The interior tail (< 16 columns) falls back to the
/// same scalar formula as the other SIMD paths.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn compute_gradients_simd128_into(img: &GrayImage, grad_x: &mut [i16], grad_y: &mut [i16]) {
    let (width_u32, height_u32) = img.dimensions();
    let width = width_u32 as usize;
    let height = height_u32 as usize;

    grad_x.fill(0);
    grad_y.fill(0);

    if width < 3 || height < 3 {
        return;
    }

    let src = img.as_raw();

    let coeff3 = i16x8_splat(3);
    let coeff10 = i16x8_splat(10);
    let interior_chunks_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let (top, mid, bottom) = unsafe {
            (
                src.as_ptr().add((y - 1) * width),
                src.as_ptr().add(y * width),
                src.as_ptr().add((y + 1) * width),
            )
        };
        let row = y * width;

        let mut x = 1usize;
        while x < interior_chunks_end {
            let (tl, tc, tr, ml, mr, bl, bc, br) = unsafe {
                (
                    load_u8x16_as_i16x8x2_simd(top.add(x - 1)),
                    load_u8x16_as_i16x8x2_simd(top.add(x)),
                    load_u8x16_as_i16x8x2_simd(top.add(x + 1)),
                    load_u8x16_as_i16x8x2_simd(mid.add(x - 1)),
                    load_u8x16_as_i16x8x2_simd(mid.add(x + 1)),
                    load_u8x16_as_i16x8x2_simd(bottom.add(x - 1)),
                    load_u8x16_as_i16x8x2_simd(bottom.add(x)),
                    load_u8x16_as_i16x8x2_simd(bottom.add(x + 1)),
                )
            };

            let gx_lo = i16x8_add(
                i16x8_mul(i16x8_sub(i16x8_add(tr.0, br.0), i16x8_add(tl.0, bl.0)), coeff3),
                i16x8_mul(i16x8_sub(mr.0, ml.0), coeff10),
            );
            let gx_hi = i16x8_add(
                i16x8_mul(i16x8_sub(i16x8_add(tr.1, br.1), i16x8_add(tl.1, bl.1)), coeff3),
                i16x8_mul(i16x8_sub(mr.1, ml.1), coeff10),
            );
            let gy_lo = i16x8_add(
                i16x8_mul(i16x8_sub(i16x8_add(bl.0, br.0), i16x8_add(tl.0, tr.0)), coeff3),
                i16x8_mul(i16x8_sub(bc.0, tc.0), coeff10),
            );
            let gy_hi = i16x8_add(
                i16x8_mul(i16x8_sub(i16x8_add(bl.1, br.1), i16x8_add(tl.1, tr.1)), coeff3),
                i16x8_mul(i16x8_sub(bc.1, tc.1), coeff10),
            );

            unsafe {
                v128_store(grad_x.as_mut_ptr().add(row + x) as *mut v128, gx_lo);
                v128_store(grad_x.as_mut_ptr().add(row + x + 8) as *mut v128, gx_hi);
                v128_store(grad_y.as_mut_ptr().add(row + x) as *mut v128, gy_lo);
                v128_store(grad_y.as_mut_ptr().add(row + x + 8) as *mut v128, gy_hi);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * width + x + 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y + 1) * width + x - 1] as i32))
                + 10 * (src[y * width + x + 1] as i32 - src[y * width + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * width + x - 1] as i32 + src[(y + 1) * width + x + 1] as i32)
                    - (src[(y - 1) * width + x - 1] as i32 + src[(y - 1) * width + x + 1] as i32))
                + 10 * (src[(y + 1) * width + x] as i32 - src[(y - 1) * width + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn load_u8x16_as_i16x8x2_simd(ptr: *const u8) -> (v128, v128) {
    let bytes = unsafe { v128_load(ptr as *const v128) };
    // Zero-extend the 16 bytes into two lanes of eight 16-bit values. The bit
    // patterns are reused directly by the signed i16x8 arithmetic above.
    (
        u16x8_extend_low_u8x16(bytes),
        u16x8_extend_high_u8x16(bytes),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocating manual reference used by the equivalence test.
    fn compute_gradients_manual(
        img: &GrayImage,
        kernel_x: &[i32; 9],
        kernel_y: &[i32; 9],
    ) -> GradientProduct {
        let (width, height) = img.dimensions();
        let mut grad_x = vec![0i16; (width * height) as usize];
        let mut grad_y = vec![0i16; (width * height) as usize];
        compute_gradients_manual_into(img, kernel_x, kernel_y, &mut grad_x, &mut grad_y);
        (
            ImageBuffer::from_vec(width, height, grad_x).unwrap(),
            ImageBuffer::from_vec(width, height, grad_y).unwrap(),
        )
    }

    fn make_test_image(width: u32, height: u32) -> GrayImage {
        let mut img = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let v = ((x * 31 + y * 17 + (x ^ y) * 13) & 0xff) as u8;
                img.put_pixel(x, y, Luma([v]));
            }
        }

        img
    }

    #[test]
    fn selected_gradients_match_manual_reference() {
        let img = make_test_image(128, 96);
        let expected =
            compute_gradients_manual(&img, &HORIZONTAL_SCHARR_3X3_OLD, &VERTICAL_SCHARR_3X3_OLD);
        let actual = compute_gradients(&img);

        assert_eq!(expected.0, actual.0, "horizontal gradients differ");
        assert_eq!(expected.1, actual.1, "vertical gradients differ");
    }

    #[test]
    fn tiny_images_return_zero_gradients() {
        for (width, height) in [(0, 0), (1, 1), (2, 2), (2, 5), (5, 2)] {
            let img = GrayImage::new(width, height);
            let (gx, gy) = compute_gradients(&img);

            assert_eq!(gx.dimensions(), (width, height));
            assert_eq!(gy.dimensions(), (width, height));
            assert!(gx.pixels().all(|p| p[0] == 0));
            assert!(gy.pixels().all(|p| p[0] == 0));
        }
    }
}
