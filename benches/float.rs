//! 8-bit versus f32 pipeline on equal workloads, with reused contexts.
//!
//! `texture_*` and `example_*` feed both pipelines the same pixel values (the
//! f32 frame holds the 8-bit values), so detection returns identical corners
//! and the timing difference is the cost of the f32 implementation. The
//! `thermal_640x480` case is the deployment comparison: a frame in °C, either
//! converted to 8 bits with a fixed 20-220 °C mapping (0.78 K per grey level,
//! `convert`) or validated as an f32 view (`validate`). Its workloads differ,
//! since quantization removes structure; counts are printed outside timing.
//!
//! Every timed call reuses a warmed context, so no allocation is timed. Image
//! generation happens before timing.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use image::{GrayImage, Luma};
use optical_flow_lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, DetectorContext, FloatDetectorContext,
    FloatImageView, FloatTrackerContext, ScoreThreshold, TrackStatus, TrackerContext,
};

#[path = "support/feature_inputs.rs"]
#[allow(dead_code)]
mod feature_inputs;
use feature_inputs::existing_points;

const QUALITY: f32 = 0.05;
const MIN_DISTANCE: u32 = 8;
const GRID: (u32, u32, u32) = (6, 4, 5);
/// f32 eigenvalue threshold equivalent to the 8-bit default on 8-bit values.
const FLOAT_MIN_EIGEN: f32 = DEFAULT_MIN_EIGEN_THRESHOLD;
/// The same threshold for a frame in °C mapped at 255 / 200 grey levels per K.
const THERMAL_MIN_EIGEN: f32 = DEFAULT_MIN_EIGEN_THRESHOLD * (200.0 / 255.0) * (200.0 / 255.0);

/// Tracking configurations: (name, levels, window, iterations, points).
const TRACKING: [(&str, usize, usize, usize, usize); 2] =
    [("pipeline", 3, 15, 15, 60), ("dense", 4, 21, 30, 300)];

struct Frame {
    name: &'static str,
    gray: GrayImage,
    celsius_or_values: Vec<f32>,
    /// Whether `gray` is the 20-220 °C mapping of `celsius_or_values`.
    thermal: bool,
}

impl Frame {
    fn equal(name: &'static str, gray: GrayImage) -> Self {
        let values = gray.as_raw().iter().map(|&v| v as f32).collect();
        Self {
            name,
            gray,
            celsius_or_values: values,
            thermal: false,
        }
    }

    fn thermal(name: &'static str, celsius: Vec<f32>, w: u32, h: u32) -> Self {
        let mut gray = GrayImage::new(w, h);
        thermal_to_gray(&celsius, &mut gray);
        Self {
            name,
            gray,
            celsius_or_values: celsius,
            thermal: true,
        }
    }

    fn view(&self) -> FloatImageView<'_> {
        let (w, h) = self.gray.dimensions();
        FloatImageView::new(&self.celsius_or_values, w, h).unwrap()
    }

    fn min_eigen(&self) -> f32 {
        if self.thermal {
            THERMAL_MIN_EIGEN
        } else {
            FLOAT_MIN_EIGEN
        }
    }
}

/// Fixed-range 8-bit mapping of a °C frame, written to vectorize.
fn thermal_to_gray(celsius: &[f32], gray: &mut GrayImage) {
    let scale = 255.0 / 200.0;
    for (out, &t) in gray.iter_mut().zip(celsius) {
        *out = ((t - 20.0) * scale).round().clamp(0.0, 255.0) as u8;
    }
}

fn hash(i: u32, seed: u32) -> f32 {
    let mut s = i
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed.wrapping_mul(40_503))
        ^ 0x9e37_79b9;
    s ^= s >> 15;
    s = s.wrapping_mul(0x85eb_ca6b);
    s ^= s >> 13;
    (s & 0xffff) as f32 / 65_535.0
}

/// A 640x480 frame in °C, content moved by `(sx, sy)`: a 22-26 °C background
/// gradient, warm bodies up to 36 °C, fine structure of a few tenths of a
/// kelvin, and 40 mK of sensor noise.
fn thermal_scene(sx: f32, sy: f32, seed: u32) -> Vec<f32> {
    let (w, h) = (640u32, 480u32);
    let bodies = [
        (150.0, 140.0, 40.0, 10.0),
        (420.0, 300.0, 70.0, 6.0),
        (560.0, 90.0, 25.0, 3.0),
        (260.0, 380.0, 55.0, 1.5),
    ];
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 - sx, (i / w) as f32 - sy);
            let mut t = 22.0 + 4.0 * y / h as f32;
            for (cx, cy, r, dt) in bodies {
                let d2 = ((x - cx).powi(2) + (y - cy).powi(2)) / (r * r);
                t += dt / (1.0 + d2 * d2);
            }
            t += 0.25 * (x * 0.21).sin() * (y * 0.17).cos() + 0.15 * ((x + 2.0 * y) * 0.05).sin();
            t + 0.04 * (hash(i, seed) - 0.5) * 3.46
        })
        .collect()
}

/// A deterministic lattice of `n` points inside a 640x480 frame.
fn lattice(n: usize) -> Vec<(f32, f32)> {
    let cols = (n as f32).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    (0..n)
        .map(|i| {
            let (c, r) = ((i % cols) as f32 + 0.5, (i / cols) as f32 + 0.5);
            (
                30.0 + c * 580.0 / cols as f32,
                30.0 + r * 420.0 / rows as f32,
            )
        })
        .collect()
}

/// `src` shifted by `(sx, sy)` with bilinear sampling, truncated to u8.
fn shift(src: &GrayImage, sx: f32, sy: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    let pixel = |x: i32, y: i32| {
        let (x, y) = (x.clamp(0, w as i32 - 1), y.clamp(0, h as i32 - 1));
        src.get_pixel(x as u32, y as u32)[0] as f32
    };
    GrayImage::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32 - sx, y as f32 - sy);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (dx, dy) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let v = pixel(x0, y0) * (1.0 - dx) * (1.0 - dy)
            + pixel(x0 + 1, y0) * dx * (1.0 - dy)
            + pixel(x0, y0 + 1) * (1.0 - dx) * dy
            + pixel(x0 + 1, y0 + 1) * dx * dy;
        Luma([v as u8])
    })
}

fn settings(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(700));
    group.measurement_time(Duration::from_secs(2));
}

fn bench_detection(c: &mut Criterion) {
    let cases = feature_inputs::cases();
    let frames = [
        Frame::equal("example_720x405", cases[0].image.clone()),
        Frame::equal("texture_640x480", cases[2].image.clone()),
        Frame::thermal("thermal_640x480", thermal_scene(0.0, 0.0, 1), 640, 480),
    ];
    let mut group = c.benchmark_group("float_vs_u8");
    settings(&mut group);
    let (cols, rows, budget) = GRID;
    let threshold = ScoreThreshold::absolute(0.0).with_relative(QUALITY);
    let mut u8_ctx = DetectorContext::new();
    let mut f32_ctx = FloatDetectorContext::new();

    for frame in &frames {
        let (w, h) = frame.gray.dimensions();
        let existing = existing_points(w, h);
        // Validation is timed separately below, as `validate`.
        let view = frame.view();
        group.throughput(Throughput::Elements(w as u64 * h as u64));

        let plain = (
            u8_ctx.detect(&frame.gray, QUALITY, MIN_DISTANCE).len(),
            f32_ctx.detect(view, threshold, MIN_DISTANCE).len(),
        );
        let top_up = (
            u8_ctx
                .detect_grid(
                    &frame.gray,
                    cols,
                    rows,
                    budget,
                    QUALITY,
                    MIN_DISTANCE,
                    &existing,
                )
                .len(),
            f32_ctx
                .detect_grid(view, cols, rows, budget, threshold, MIN_DISTANCE, &existing)
                .len(),
        );
        eprintln!(
            "float_vs_u8: case={}, corners plain u8={} f32={}, grid_top_up u8={} f32={}",
            frame.name, plain.0, plain.1, top_up.0, top_up.1
        );

        group.bench_function(BenchmarkId::new("detect_plain/u8", frame.name), |b| {
            b.iter(|| {
                black_box(
                    u8_ctx
                        .detect(black_box(&frame.gray), QUALITY, MIN_DISTANCE)
                        .len(),
                )
            })
        });
        group.bench_function(BenchmarkId::new("detect_plain/f32", frame.name), |b| {
            b.iter(|| {
                black_box(
                    f32_ctx
                        .detect(black_box(view), threshold, MIN_DISTANCE)
                        .len(),
                )
            })
        });
        group.bench_function(BenchmarkId::new("detect_grid_top_up/u8", frame.name), |b| {
            b.iter(|| {
                let image = black_box(&frame.gray);
                black_box(
                    u8_ctx
                        .detect_grid(image, cols, rows, budget, QUALITY, MIN_DISTANCE, &existing)
                        .len(),
                )
            })
        });
        group.bench_function(
            BenchmarkId::new("detect_grid_top_up/f32", frame.name),
            |b| {
                b.iter(|| {
                    let image = black_box(view);
                    black_box(
                        f32_ctx
                            .detect_grid(
                                image,
                                cols,
                                rows,
                                budget,
                                threshold,
                                MIN_DISTANCE,
                                &existing,
                            )
                            .len(),
                    )
                })
            },
        );

        if frame.thermal {
            // Per-frame input preparation: 8-bit conversion versus f32 validation.
            let mut gray = frame.gray.clone();
            group.bench_function(BenchmarkId::new("convert/u8", frame.name), |b| {
                b.iter(|| thermal_to_gray(black_box(&frame.celsius_or_values), &mut gray))
            });
            group.bench_function(BenchmarkId::new("validate/f32", frame.name), |b| {
                b.iter(|| {
                    black_box(
                        FloatImageView::new(black_box(&frame.celsius_or_values), w, h).is_ok(),
                    )
                })
            });
        }
    }
    group.finish();
}

fn bench_tracking(c: &mut Criterion) {
    let texture = feature_inputs::texture(640, 480);
    let pairs = [
        (
            Frame::equal("texture_640x480", texture.clone()),
            Frame::equal("texture_640x480", shift(&texture, 3.0, 2.0)),
        ),
        (
            Frame::thermal("thermal_640x480", thermal_scene(0.0, 0.0, 1), 640, 480),
            Frame::thermal("thermal_640x480", thermal_scene(3.0, 2.0, 2), 640, 480),
        ),
    ];
    let mut group = c.benchmark_group("float_vs_u8");
    settings(&mut group);
    let mut u8_ctx = TrackerContext::new();
    let mut f32_ctx = FloatTrackerContext::new();
    let mut u8_detector = DetectorContext::new();
    let mut f32_detector = FloatDetectorContext::new();
    let (cols, rows, budget) = GRID;
    let threshold = ScoreThreshold::absolute(0.0).with_relative(QUALITY);

    for (prev, next) in &pairs {
        let name = prev.name;
        let (prev_view, next_view) = (prev.view(), next.view());
        for (config, levels, window, iterations, n) in TRACKING {
            let points = lattice(n);
            let min_eigen = prev.min_eigen();
            u8_ctx.prepare(&prev.gray, &next.gray, levels);
            f32_ctx.prepare(prev_view, next_view, levels);
            // Tracked count and mean position error against the true (3, 2) px
            // motion, outside timing.
            let quality = |results: &[optical_flow_lk::TrackResult]| {
                let errors: Vec<f32> = results
                    .iter()
                    .zip(&points)
                    .filter(|(r, _)| r.status == TrackStatus::Tracked)
                    .map(|(r, &(x, y))| {
                        ((r.pos.0 - x - 3.0).powi(2) + (r.pos.1 - y - 2.0).powi(2)).sqrt()
                    })
                    .collect();
                let mean = errors.iter().sum::<f32>() / errors.len().max(1) as f32;
                format!("{} ({mean:.3} px)", errors.len())
            };
            let u8_fb = quality(u8_ctx.track_fb(
                &points,
                None,
                window,
                iterations,
                DEFAULT_MIN_EIGEN_THRESHOLD,
                DEFAULT_FB_THRESHOLD,
            ));
            let f32_fb = quality(f32_ctx.track_fb(
                &points,
                None,
                window,
                iterations,
                min_eigen,
                DEFAULT_FB_THRESHOLD,
            ));
            eprintln!(
                "float_vs_u8: case={name}, tracking={config}, points={n}, tracked_fb u8={u8_fb} f32={f32_fb}"
            );
            let id = |stage: &str, kind: &str| {
                BenchmarkId::new(format!("{stage}_{config}/{kind}"), name)
            };

            group.bench_function(id("prepare", "u8"), |b| {
                b.iter(|| u8_ctx.prepare(black_box(&prev.gray), black_box(&next.gray), levels))
            });
            group.bench_function(id("prepare", "f32"), |b| {
                b.iter(|| f32_ctx.prepare(black_box(prev_view), black_box(next_view), levels))
            });
            u8_ctx.prepare(&prev.gray, &next.gray, levels);
            f32_ctx.prepare(prev_view, next_view, levels);
            group.bench_function(id("track", "u8"), |b| {
                b.iter(|| {
                    black_box(
                        u8_ctx
                            .track(
                                black_box(&points),
                                None,
                                window,
                                iterations,
                                DEFAULT_MIN_EIGEN_THRESHOLD,
                            )
                            .len(),
                    )
                })
            });
            group.bench_function(id("track", "f32"), |b| {
                b.iter(|| {
                    black_box(
                        f32_ctx
                            .track(black_box(&points), None, window, iterations, min_eigen)
                            .len(),
                    )
                })
            });
            group.bench_function(id("track_fb", "u8"), |b| {
                b.iter(|| {
                    black_box(
                        u8_ctx
                            .track_fb(
                                black_box(&points),
                                None,
                                window,
                                iterations,
                                DEFAULT_MIN_EIGEN_THRESHOLD,
                                DEFAULT_FB_THRESHOLD,
                            )
                            .len(),
                    )
                })
            });
            group.bench_function(id("track_fb", "f32"), |b| {
                b.iter(|| {
                    black_box(
                        f32_ctx
                            .track_fb(
                                black_box(&points),
                                None,
                                window,
                                iterations,
                                min_eigen,
                                DEFAULT_FB_THRESHOLD,
                            )
                            .len(),
                    )
                })
            });

            if config != "pipeline" {
                continue;
            }
            // One frame of a tracking front end: prepare the new frame (8-bit
            // conversion or f32 validation for thermal input), build both
            // pyramids, track with the forward-backward check, and top up the
            // grid around the surviving points.
            let mut gray = next.gray.clone();
            let mut survivors = Vec::with_capacity(n);
            group.bench_function(id("frame_step", "u8"), |b| {
                b.iter(|| {
                    if next.thermal {
                        thermal_to_gray(black_box(&next.celsius_or_values), &mut gray);
                    }
                    u8_ctx.prepare(&prev.gray, &gray, levels);
                    let results = u8_ctx.track_fb(
                        &points,
                        None,
                        window,
                        iterations,
                        DEFAULT_MIN_EIGEN_THRESHOLD,
                        DEFAULT_FB_THRESHOLD,
                    );
                    survivors.clear();
                    survivors.extend(
                        results
                            .iter()
                            .filter(|r| r.status == TrackStatus::Tracked)
                            .map(|r| r.pos),
                    );
                    black_box(
                        u8_detector
                            .detect_grid(
                                &gray,
                                cols,
                                rows,
                                budget,
                                QUALITY,
                                MIN_DISTANCE,
                                &survivors,
                            )
                            .len(),
                    )
                })
            });
            group.bench_function(id("frame_step", "f32"), |b| {
                b.iter(|| {
                    // The previous frame was validated one step earlier.
                    let (w, h) = next.gray.dimensions();
                    let next_view =
                        FloatImageView::new(black_box(&next.celsius_or_values), w, h).unwrap();
                    f32_ctx.prepare(prev_view, next_view, levels);
                    let results = f32_ctx.track_fb(
                        &points,
                        None,
                        window,
                        iterations,
                        min_eigen,
                        DEFAULT_FB_THRESHOLD,
                    );
                    survivors.clear();
                    survivors.extend(
                        results
                            .iter()
                            .filter(|r| r.status == TrackStatus::Tracked)
                            .map(|r| r.pos),
                    );
                    black_box(
                        f32_detector
                            .detect_grid(
                                next_view,
                                cols,
                                rows,
                                budget,
                                threshold,
                                MIN_DISTANCE,
                                &survivors,
                            )
                            .len(),
                    )
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_detection, bench_tracking);
criterion_main!(benches);
