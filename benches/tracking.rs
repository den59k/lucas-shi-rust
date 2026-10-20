//! Criterion bench for the per-frame track step at 640x480.
//!
//! `track_step_640x480` compares the allocating free-function path ("before"
//! the zero-alloc refactor: fresh pyramids + fresh scratch every frame) against
//! the buffer-reusing `TrackerContext` path ("after"), across 50/150/300 points
//! and with/without an initial guess.
//!
//! `frame_step_640x480` compares the two per-frame forms of `TrackerContext`:
//! `prepare(last, frame)`, which builds both pyramids, against
//! `advance(frame)`, which reuses the last frame's pyramid. Each iteration is
//! one new frame, alternating between two frames; `pyramids` times the pyramid
//! step alone, `track_fb/150` adds forward-backward tracking of 150 points.
//!
//! `regime` is the per-frame step of a real-time front-end: `advance` with the
//! new frame, then `track` (`fb_off`) or `track_fb` (`fb_all`) of the
//! strongest corners of the first frame. Portrait 360x640 and 640x480 frames,
//! 4 levels, windows 13 and 21, 44 and 150 points; ids are
//! `regime/<size>/w<window>/p<points>/<fb>`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

use image::{GrayImage, Luma};
use optical_flow_lk::{
    Corner, DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, TrackParams, TrackerContext,
    build_pyramid, calc_optical_flow_ex, good_features_to_track,
};

const W: u32 = 640;
const H: u32 = 480;
const LEVELS: usize = 4;
const WIN: usize = 21;
const ITERS: usize = 30;
const PARAMS: TrackParams = TrackParams {
    window_size: WIN,
    max_iterations: ITERS,
    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
};

fn textured(seed: u32) -> GrayImage {
    textured_sized(W, H, seed)
}

/// Hash noise band-limited by two 3x3 means, `w` x `h`.
fn textured_sized(w: u32, h: u32, seed: u32) -> GrayImage {
    let mut buf: Vec<f32> = (0..(w * h))
        .map(|i| {
            let mut s = i
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503))
                ^ 0x9e3779b9;
            s ^= s >> 15;
            s = s.wrapping_mul(0x85ebca6b);
            s ^= s >> 13;
            (s & 0xff) as f32
        })
        .collect();
    // Two light box blurs to band-limit the texture.
    for _ in 0..2 {
        let src = buf.clone();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (mut sum, mut n) = (0.0, 0.0);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                            sum += src[(ny as u32 * w + nx as u32) as usize];
                            n += 1.0;
                        }
                    }
                }
                buf[(y as u32 * w + x as u32) as usize] = sum / n;
            }
        }
    }
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(x, y, Luma([buf[(y * w + x) as usize] as u8]));
        }
    }
    img
}

/// A deterministic lattice of `n` in-bounds points.
fn lattice(n: usize) -> Vec<(f32, f32)> {
    let cols = (n as f32).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let (cx, cy) = (i % cols, i / cols);
        let x = 30.0 + (cx as f32 + 0.5) * (W as f32 - 60.0) / cols as f32;
        let y = 30.0 + (cy as f32 + 0.5) * (H as f32 - 60.0) / rows as f32;
        pts.push((x, y));
    }
    pts
}

fn bench_tracking(c: &mut Criterion) {
    let prev = textured(1);
    // ~3 px global motion is representative of a calm hand-held frame pair.
    let next = shift(&prev, 3.0, 2.0);

    let mut group = c.benchmark_group("track_step_640x480");

    for &n in &[50usize, 150, 300] {
        let points = lattice(n);
        let predicted: Vec<(f32, f32)> = points.iter().map(|&(x, y)| (x + 3.0, y + 2.0)).collect();

        for &guess in &[false, true] {
            let guess_opt = if guess {
                Some(predicted.as_slice())
            } else {
                None
            };
            let tag = if guess { "guess" } else { "noguess" };

            // "before": allocate fresh pyramids and scratch every frame.
            group.bench_with_input(BenchmarkId::new(format!("before/{tag}"), n), &n, |b, _| {
                b.iter(|| {
                    let pp = build_pyramid(&prev, LEVELS);
                    let np = build_pyramid(&next, LEVELS);
                    black_box(calc_optical_flow_ex(&pp, &np, &points, guess_opt, PARAMS))
                });
            });

            // "after": reuse one warm TrackerContext.
            let mut ctx = TrackerContext::new();
            ctx.prepare(&prev, &next, LEVELS);
            ctx.track(&points, guess_opt, PARAMS);
            group.bench_with_input(BenchmarkId::new(format!("after/{tag}"), n), &n, |b, _| {
                b.iter(|| {
                    ctx.prepare(&prev, &next, LEVELS);
                    black_box(ctx.track(&points, guess_opt, PARAMS));
                });
            });
        }
    }

    group.finish();
}

fn bench_frame_step(c: &mut Criterion) {
    let first = textured(1);
    let frames = [first.clone(), shift(&first, 3.0, 2.0)];
    let points = lattice(150);

    let mut group = c.benchmark_group("frame_step_640x480");
    for tracking in [false, true] {
        let case = if tracking { "track_fb/150" } else { "pyramids" };

        let mut ctx = TrackerContext::new();
        let mut index = 0;
        group.bench_function(BenchmarkId::new("prepare", case), |b| {
            b.iter(|| {
                index ^= 1;
                ctx.prepare(&frames[index ^ 1], &frames[index], LEVELS);
                if tracking {
                    black_box(ctx.track_fb(&points, None, PARAMS, DEFAULT_FB_THRESHOLD));
                }
                black_box(ctx.next_pyramid());
            });
        });

        let mut ctx = TrackerContext::new();
        ctx.advance(&frames[0], LEVELS);
        let mut index = 0;
        group.bench_function(BenchmarkId::new("advance", case), |b| {
            b.iter(|| {
                index ^= 1;
                ctx.advance(&frames[index], LEVELS);
                if tracking {
                    black_box(ctx.track_fb(&points, None, PARAMS, DEFAULT_FB_THRESHOLD));
                }
                black_box(ctx.next_pyramid());
            });
        });
    }
    group.finish();
}

/// Forward-backward mode of a `regime` case.
#[derive(Clone, Copy)]
enum Fb {
    Off,
    All,
}

impl Fb {
    fn label(self) -> &'static str {
        match self {
            Fb::Off => "fb_off",
            Fb::All => "fb_all",
        }
    }
}

fn bench_regime(c: &mut Criterion) {
    let mut group = c.benchmark_group("regime");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for (w, h) in [(360u32, 640u32), (640, 480)] {
        let first = textured_sized(w, h, 1);
        let frames = [first.clone(), shift(&first, 3.0, 2.0)];
        let corners = good_features_to_track(&frames[0], 0.01, 10);
        for n in [44usize, 150] {
            let points: Vec<(f32, f32)> = corners.iter().take(n).map(Corner::position).collect();
            assert_eq!(points.len(), n, "too few corners at {w}x{h}");
            for window in [13usize, 21] {
                let params = TrackParams {
                    window_size: window,
                    ..PARAMS
                };
                for fb in [Fb::Off, Fb::All] {
                    let mut ctx = TrackerContext::new();
                    ctx.advance(&frames[0], LEVELS);
                    let mut index = 0;
                    let id = format!("{w}x{h}/w{window}/p{n}/{}", fb.label());
                    group.bench_function(id, |b| {
                        b.iter(|| {
                            index ^= 1;
                            ctx.advance(&frames[index], LEVELS);
                            black_box(match fb {
                                Fb::Off => ctx.track(&points, None, params),
                                Fb::All => {
                                    ctx.track_fb(&points, None, params, DEFAULT_FB_THRESHOLD)
                                }
                            });
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

/// Local shift helper (bench has no access to the test helpers).
fn shift(src: &GrayImage, sx: f32, sy: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    let mut out = GrayImage::new(w, h);
    let sample = |x: f32, y: f32| -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let (dx, dy) = (x - x0 as f32, y - y0 as f32);
        let g = |xx: i32, yy: i32| {
            if xx >= 0 && yy >= 0 && xx < w as i32 && yy < h as i32 {
                src.get_pixel(xx as u32, yy as u32)[0] as f32
            } else {
                0.0
            }
        };
        g(x0, y0) * (1.0 - dx) * (1.0 - dy)
            + g(x0 + 1, y0) * dx * (1.0 - dy)
            + g(x0, y0 + 1) * (1.0 - dx) * dy
            + g(x0 + 1, y0 + 1) * dx * dy
    };
    for y in 0..h {
        for x in 0..w {
            out.put_pixel(x, y, Luma([sample(x as f32 - sx, y as f32 - sy) as u8]));
        }
    }
    out
}

criterion_group!(benches, bench_tracking, bench_frame_step, bench_regime);
criterion_main!(benches);
