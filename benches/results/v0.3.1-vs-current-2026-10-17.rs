//! 0.3.1 versus the current revision on identical workloads. Built twice from
//! this file: against 0.3.1 (common 8-bit API only) and against the current
//! revision with the `current` feature (contexts and the f32 pipeline).

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use image::{GrayImage, Luma};
use optical_flow_lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, TrackResult, TrackStatus, TrackerContext,
    good_features_to_track, good_features_to_track_grid,
};
#[cfg(feature = "current")]
use optical_flow_lk::{
    DetectorContext, FloatDetectorContext, FloatImageView, FloatTrackerContext, ScoreThreshold,
};

// Point this at benches/support/feature_inputs.rs of this repository.
#[path = "/path/to/lucas-shi-rust/benches/support/feature_inputs.rs"]
#[allow(dead_code)]
mod feature_inputs;
use feature_inputs::existing_points;

const QUALITY: f32 = 0.05;
const MIN_DISTANCE: u32 = 8;
const GRID: (u32, u32, u32) = (6, 4, 5);
const THERMAL_MIN_EIGEN: f32 = DEFAULT_MIN_EIGEN_THRESHOLD * (200.0 / 255.0) * (200.0 / 255.0);
const TRACKING: [(&str, usize, usize, usize, usize); 2] =
    [("pipeline", 3, 15, 15, 60), ("dense", 4, 21, 30, 300)];

#[cfg(feature = "current")]
const REVISION: &str = "current";
#[cfg(not(feature = "current"))]
const REVISION: &str = "v0.3.1";

struct Frame {
    name: &'static str,
    gray: GrayImage,
    values: Vec<f32>,
    thermal: bool,
}

impl Frame {
    fn equal(name: &'static str, gray: GrayImage) -> Self {
        let values = gray.as_raw().iter().map(|&v| v as f32).collect();
        Self { name, gray, values, thermal: false }
    }

    fn thermal(name: &'static str, celsius: Vec<f32>) -> Self {
        let mut gray = GrayImage::new(640, 480);
        thermal_to_gray(&celsius, &mut gray);
        Self { name, gray, values: celsius, thermal: true }
    }

    #[cfg(feature = "current")]
    fn view(&self) -> FloatImageView<'_> {
        let (w, h) = self.gray.dimensions();
        FloatImageView::new(&self.values, w, h).unwrap()
    }

    #[allow(dead_code)]
    fn min_eigen(&self) -> f32 {
        if self.thermal { THERMAL_MIN_EIGEN } else { DEFAULT_MIN_EIGEN_THRESHOLD }
    }
}

fn thermal_to_gray(celsius: &[f32], gray: &mut GrayImage) {
    let scale = 255.0 / 200.0;
    for (out, &t) in gray.iter_mut().zip(celsius) {
        *out = ((t - 20.0) * scale).round().clamp(0.0, 255.0) as u8;
    }
}

fn hash(i: u32, seed: u32) -> f32 {
    let mut s = i.wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40_503)) ^ 0x9e37_79b9;
    s ^= s >> 15;
    s = s.wrapping_mul(0x85eb_ca6b);
    s ^= s >> 13;
    (s & 0xffff) as f32 / 65_535.0
}

/// Same scene as benches/float.rs.
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

fn lattice(n: usize) -> Vec<(f32, f32)> {
    let cols = (n as f32).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    (0..n)
        .map(|i| {
            let (c, r) = ((i % cols) as f32 + 0.5, (i / cols) as f32 + 0.5);
            (30.0 + c * 580.0 / cols as f32, 30.0 + r * 420.0 / rows as f32)
        })
        .collect()
}

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

fn quality(results: &[TrackResult], points: &[(f32, f32)]) -> String {
    let errors: Vec<f32> = results
        .iter()
        .zip(points)
        .filter(|(r, _)| r.status == TrackStatus::Tracked)
        .map(|(r, &(x, y))| ((r.pos.0 - x - 3.0).powi(2) + (r.pos.1 - y - 2.0).powi(2)).sqrt())
        .collect();
    let mean = errors.iter().sum::<f32>() / errors.len().max(1) as f32;
    format!("{} ({mean:.3} px)", errors.len())
}

fn settings(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_millis(1500));
}

fn bench_detection(c: &mut Criterion) {
    let mut frames: Vec<Frame> = feature_inputs::cases()
        .into_iter()
        .map(|case| Frame::equal(case.name, case.image))
        .collect();
    frames.push(Frame::thermal("thermal_640x480", thermal_scene(0.0, 0.0, 1)));
    let mut group = c.benchmark_group("v031");
    settings(&mut group);
    let (cols, rows, budget) = GRID;

    for frame in &frames {
        let (w, h) = frame.gray.dimensions();
        let existing = existing_points(w, h);
        let image = &frame.gray;
        eprintln!(
            "v031: revision={REVISION}, case={}, corners u8 plain={} grid_top_up={}",
            frame.name,
            good_features_to_track(image, QUALITY, MIN_DISTANCE).len(),
            good_features_to_track_grid(image, cols, rows, budget, QUALITY, MIN_DISTANCE, &existing).len()
        );
        group.bench_function(BenchmarkId::new(format!("detect_plain/u8_free/{REVISION}"), frame.name), |b| {
            b.iter(|| black_box(good_features_to_track(black_box(image), QUALITY, MIN_DISTANCE).len()))
        });
        group.bench_function(BenchmarkId::new(format!("detect_grid_top_up/u8_free/{REVISION}"), frame.name), |b| {
            b.iter(|| {
                black_box(
                    good_features_to_track_grid(black_box(image), cols, rows, budget, QUALITY, MIN_DISTANCE, &existing)
                        .len(),
                )
            })
        });

        #[cfg(feature = "current")]
        {
            let view = frame.view();
            let threshold = ScoreThreshold::absolute(0.0).with_relative(QUALITY);
            let mut u8_ctx = DetectorContext::new();
            let mut f32_ctx = FloatDetectorContext::new();
            eprintln!(
                "v031: revision={REVISION}, case={}, corners f32 plain={} grid_top_up={}",
                frame.name,
                f32_ctx.detect(view, threshold, MIN_DISTANCE).len(),
                f32_ctx.detect_grid(view, cols, rows, budget, threshold, MIN_DISTANCE, &existing).len()
            );
            u8_ctx.detect(image, QUALITY, MIN_DISTANCE);
            group.bench_function(BenchmarkId::new("detect_plain/u8_context/current", frame.name), |b| {
                b.iter(|| black_box(u8_ctx.detect(black_box(image), QUALITY, MIN_DISTANCE).len()))
            });
            // A fresh context per call allocates like the free functions do.
            group.bench_function(BenchmarkId::new("detect_plain/f32_fresh/current", frame.name), |b| {
                b.iter(|| {
                    let mut ctx = FloatDetectorContext::new();
                    black_box(ctx.detect(black_box(view), threshold, MIN_DISTANCE).len())
                })
            });
            group.bench_function(BenchmarkId::new("detect_plain/f32_context/current", frame.name), |b| {
                b.iter(|| black_box(f32_ctx.detect(black_box(view), threshold, MIN_DISTANCE).len()))
            });
            group.bench_function(BenchmarkId::new("detect_grid_top_up/u8_context/current", frame.name), |b| {
                b.iter(|| {
                    black_box(u8_ctx.detect_grid(black_box(image), cols, rows, budget, QUALITY, MIN_DISTANCE, &existing).len())
                })
            });
            group.bench_function(BenchmarkId::new("detect_grid_top_up/f32_fresh/current", frame.name), |b| {
                b.iter(|| {
                    let mut ctx = FloatDetectorContext::new();
                    black_box(ctx.detect_grid(black_box(view), cols, rows, budget, threshold, MIN_DISTANCE, &existing).len())
                })
            });
            group.bench_function(BenchmarkId::new("detect_grid_top_up/f32_context/current", frame.name), |b| {
                b.iter(|| {
                    black_box(f32_ctx.detect_grid(black_box(view), cols, rows, budget, threshold, MIN_DISTANCE, &existing).len())
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
            Frame::thermal("thermal_640x480", thermal_scene(0.0, 0.0, 1)),
            Frame::thermal("thermal_640x480", thermal_scene(3.0, 2.0, 2)),
        ),
    ];
    let mut group = c.benchmark_group("v031");
    settings(&mut group);
    let (cols, rows, budget) = GRID;
    let mut u8_ctx = TrackerContext::new();

    for (prev, next) in &pairs {
        let name = prev.name;
        for (config, levels, window, iterations, n) in TRACKING {
            let points = lattice(n);
            let thr = DEFAULT_MIN_EIGEN_THRESHOLD;
            let fb = DEFAULT_FB_THRESHOLD;
            u8_ctx.prepare(&prev.gray, &next.gray, levels);
            eprintln!(
                "v031: revision={REVISION}, case={name}, tracking={config}, u8 tracked_fb={}",
                quality(u8_ctx.track_fb(&points, None, window, iterations, thr, fb), &points)
            );
            let id = |stage: &str, kind: &str| BenchmarkId::new(format!("{stage}_{config}/{kind}"), name);
            group.bench_function(id("prepare", &format!("u8/{REVISION}")), |b| {
                b.iter(|| u8_ctx.prepare(black_box(&prev.gray), black_box(&next.gray), levels))
            });
            u8_ctx.prepare(&prev.gray, &next.gray, levels);
            group.bench_function(id("track", &format!("u8/{REVISION}")), |b| {
                b.iter(|| black_box(u8_ctx.track(black_box(&points), None, window, iterations, thr).len()))
            });
            group.bench_function(id("track_fb", &format!("u8/{REVISION}")), |b| {
                b.iter(|| black_box(u8_ctx.track_fb(black_box(&points), None, window, iterations, thr, fb).len()))
            });

            #[cfg(feature = "current")]
            let mut f32_ctx = FloatTrackerContext::new();
            #[cfg(feature = "current")]
            {
                let (pv, nv) = (prev.view(), next.view());
                let me = prev.min_eigen();
                f32_ctx.prepare(pv, nv, levels);
                eprintln!(
                    "v031: revision={REVISION}, case={name}, tracking={config}, f32 tracked_fb={}",
                    quality(f32_ctx.track_fb(&points, None, window, iterations, me, fb), &points)
                );
                group.bench_function(id("prepare", "f32/current"), |b| {
                    b.iter(|| f32_ctx.prepare(black_box(pv), black_box(nv), levels))
                });
                f32_ctx.prepare(pv, nv, levels);
                group.bench_function(id("track", "f32/current"), |b| {
                    b.iter(|| black_box(f32_ctx.track(black_box(&points), None, window, iterations, me).len()))
                });
                group.bench_function(id("track_fb", "f32/current"), |b| {
                    b.iter(|| black_box(f32_ctx.track_fb(black_box(&points), None, window, iterations, me, fb).len()))
                });
            }

            if config != "pipeline" {
                continue;
            }
            // Frame step: prepare the new frame, both pyramids, track_fb,
            // grid top-up around the survivors.
            let mut gray = next.gray.clone();
            let mut survivors: Vec<(f32, f32)> = Vec::with_capacity(n);
            group.bench_function(id("frame_step", &format!("u8_free/{REVISION}")), |b| {
                b.iter(|| {
                    if next.thermal {
                        thermal_to_gray(black_box(&next.values), &mut gray);
                    }
                    u8_ctx.prepare(&prev.gray, &gray, levels);
                    let results = u8_ctx.track_fb(&points, None, window, iterations, thr, fb);
                    survivors.clear();
                    survivors.extend(results.iter().filter(|r| r.status == TrackStatus::Tracked).map(|r| r.pos));
                    black_box(good_features_to_track_grid(&gray, cols, rows, budget, QUALITY, MIN_DISTANCE, &survivors).len())
                })
            });
            #[cfg(feature = "current")]
            {
                let mut u8_detector = DetectorContext::new();
                let mut f32_detector = FloatDetectorContext::new();
                let threshold = ScoreThreshold::absolute(0.0).with_relative(QUALITY);
                let me = prev.min_eigen();
                let pv = prev.view();
                group.bench_function(id("frame_step", "u8_context/current"), |b| {
                    b.iter(|| {
                        if next.thermal {
                            thermal_to_gray(black_box(&next.values), &mut gray);
                        }
                        u8_ctx.prepare(&prev.gray, &gray, levels);
                        let results = u8_ctx.track_fb(&points, None, window, iterations, thr, fb);
                        survivors.clear();
                        survivors.extend(results.iter().filter(|r| r.status == TrackStatus::Tracked).map(|r| r.pos));
                        black_box(u8_detector.detect_grid(&gray, cols, rows, budget, QUALITY, MIN_DISTANCE, &survivors).len())
                    })
                });
                group.bench_function(id("frame_step", "f32_context/current"), |b| {
                    b.iter(|| {
                        let nv = FloatImageView::new(black_box(&next.values), 640, 480).unwrap();
                        f32_ctx.prepare(pv, nv, levels);
                        let results = f32_ctx.track_fb(&points, None, window, iterations, me, fb);
                        survivors.clear();
                        survivors.extend(results.iter().filter(|r| r.status == TrackStatus::Tracked).map(|r| r.pos));
                        black_box(f32_detector.detect_grid(nv, cols, rows, budget, threshold, MIN_DISTANCE, &survivors).len())
                    })
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_detection, bench_tracking);
criterion_main!(benches);
