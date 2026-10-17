//! Whole-call Shi–Tomasi benchmarks. Build this same harness against each
//! revision to compare numeric fixes; no historical detector is embedded here.
//!
//! Image generation/decoding, existing points, and corner-count reporting happen
//! outside timing. Each timed call includes detection, NMS, sorting, spacing,
//! grid occupancy where applicable, and allocation/deallocation of its buffers.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use image::{GrayImage, Luma};
use optical_flow_lk::{good_features_to_track, good_features_to_track_grid};

const QUALITY: f32 = 0.05;
const MIN_DISTANCE: u32 = 8;
const GRID_COLS: u32 = 6;
const GRID_ROWS: u32 = 4;
const MAX_PER_CELL: u32 = 5;

struct Case {
    name: &'static str,
    image: GrayImage,
}

#[derive(Clone, Copy)]
enum Mode {
    Plain,
    Grid,
    GridTopUp,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Grid => "grid",
            Self::GridTopUp => "grid_top_up",
        }
    }

    fn detect(self, image: &GrayImage, existing: &[(f32, f32)]) -> Vec<(u32, u32, f32)> {
        match self {
            Self::Plain => good_features_to_track(image, QUALITY, MIN_DISTANCE),
            Self::Grid | Self::GridTopUp => good_features_to_track_grid(
                image,
                GRID_COLS,
                GRID_ROWS,
                MAX_PER_CELL,
                QUALITY,
                MIN_DISTANCE,
                existing,
            ),
        }
    }
}

/// Seeded texture, band-limited by two 3x3 averages. Arithmetic and input bytes
/// are fixed across revisions; no random generator state or clock is involved.
fn texture(width: u32, height: u32) -> GrayImage {
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
fn existing_points(width: u32, height: u32) -> Vec<(f32, f32)> {
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

fn bench_features(c: &mut Criterion) {
    let cases = [
        Case {
            name: "example_720x405",
            image: image::load_from_memory(include_bytes!("../examples/input1.png"))
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
            name: "flat_640x480",
            image: GrayImage::from_pixel(640, 480, Luma([127])),
        },
    ];
    eprintln!(
        "features: quality={QUALITY}, min_distance={MIN_DISTANCE}px, \
         grid={GRID_COLS}x{GRID_ROWS}, max_per_cell={MAX_PER_CELL}"
    );
    let mut group = c.benchmark_group("features");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for case in &cases {
        let (width, height) = case.image.dimensions();
        let existing = existing_points(width, height);
        group.throughput(Throughput::Elements(width as u64 * height as u64));
        for mode in [Mode::Plain, Mode::Grid, Mode::GridTopUp] {
            let occupied = match mode {
                Mode::GridTopUp => existing.as_slice(),
                Mode::Plain | Mode::Grid => &[],
            };
            let count = mode.detect(&case.image, occupied).len();
            eprintln!(
                "features: case={}, api={}, corners={count}, existing={}",
                case.name,
                mode.name(),
                occupied.len()
            );
            group.bench_with_input(
                BenchmarkId::new(mode.name(), case.name),
                &case.image,
                |b, image| {
                    b.iter(|| {
                        black_box(mode.detect(black_box(image), black_box(occupied)));
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_features);
criterion_main!(benches);
