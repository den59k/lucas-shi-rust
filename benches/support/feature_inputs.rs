//! Deterministic inputs shared by whole-call and stage benchmarks.

use image::{GrayImage, Luma};

const GRID_COLS: u32 = 6;
const GRID_ROWS: u32 = 4;

pub struct Case {
    pub name: &'static str,
    pub image: GrayImage,
}

/// Seeded texture, band-limited by two 3x3 averages. Arithmetic and input bytes
/// are fixed across revisions; no random generator state or clock is involved.
pub fn texture(width: u32, height: u32) -> GrayImage {
    let mut pixels: Vec<f32> = (0..width * height)
        .map(|i| {
            let mut state = i.wrapping_mul(2_654_435_761).wrapping_add(40_503) ^ 0x9e37_79b9;
            state ^= state >> 15;
            state = state.wrapping_mul(0x85eb_ca6b);
            state ^= state >> 13;
            (state & 255) as f32
        })
        .collect();
    for _ in 0..2 {
        let source = pixels.clone();
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut count = 0;
                for sy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                    for sx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        sum += source[(sy * width + sx) as usize];
                        count += 1;
                    }
                }
                pixels[(y * width + x) as usize] = sum / count as f32;
            }
        }
    }
    GrayImage::from_fn(width, height, |x, y| {
        Luma([pixels[(y * width + x) as usize] as u8])
    })
}

/// One existing point at the center of each detection cell, independent of the
/// detector revision. This reserves one of each cell's five available slots.
pub fn existing_points(width: u32, height: u32) -> Vec<(f32, f32)> {
    (0..GRID_ROWS)
        .flat_map(|row| {
            (0..GRID_COLS).map(move |col| {
                (
                    (col as f32 + 0.5) * width as f32 / GRID_COLS as f32,
                    (row as f32 + 0.5) * height as f32 / GRID_ROWS as f32,
                )
            })
        })
        .collect()
}

pub fn cases() -> [Case; 5] {
    [
        Case {
            name: "example_720x405",
            image: image::load_from_memory(include_bytes!("../../examples/input1.png"))
                .expect("bundled example image must decode")
                .into_luma8(),
        },
        Case {
            name: "texture_320x240",
            image: texture(320, 240),
        },
        Case {
            name: "texture_640x480",
            image: texture(640, 480),
        },
        Case {
            name: "low_contrast_320x240",
            image: {
                let mut image = texture(320, 240);
                for pixel in image.pixels_mut() {
                    pixel[0] /= 64;
                }
                image
            },
        },
        Case {
            name: "flat_640x480",
            image: GrayImage::from_pixel(640, 480, Luma([127])),
        },
    ]
}
