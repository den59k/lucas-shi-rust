//! Whole-call Shi–Tomasi benchmarks. Build this same harness against each
//! revision to compare numeric fixes; no historical detector is embedded here.
//!
//! Image generation/decoding, existing points, and corner-count reporting happen
//! outside timing. Each timed call includes detection, NMS, sorting, spacing,
//! grid occupancy where applicable, and allocation/deallocation of its buffers.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use image::GrayImage;

#[path = "support/feature_inputs.rs"]
mod feature_inputs;
use feature_inputs::existing_points;
use optical_flow_lk::{Corner, Grid, good_features_to_track, good_features_to_track_grid};

const QUALITY: f32 = 0.05;
const MIN_DISTANCE: u32 = 8;
const GRID_COLS: u32 = 6;
const GRID_ROWS: u32 = 4;
const MAX_PER_CELL: u32 = 5;

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

    fn detect(self, image: &GrayImage, existing: &[(f32, f32)]) -> Vec<Corner> {
        match self {
            Self::Plain => good_features_to_track(image, QUALITY, MIN_DISTANCE),
            Self::Grid | Self::GridTopUp => good_features_to_track_grid(
                image,
                Grid {
                    cols: GRID_COLS,
                    rows: GRID_ROWS,
                    max_per_cell: MAX_PER_CELL,
                },
                QUALITY,
                MIN_DISTANCE,
                existing,
            ),
        }
    }
}

fn bench_features(c: &mut Criterion) {
    let cases = feature_inputs::cases();
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
