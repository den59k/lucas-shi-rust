//! VIO front-end walkthrough on a purely synthetic 3-frame sequence:
//!
//!   grid detection  ->  predicted-guess tracking  ->  forward-backward check
//!
//! Run with `cargo run --example vio_frontend`. No image files are required.

use image::{GrayImage, Luma};
use optical_flow_lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, TrackStatus, TrackerContext,
    good_features_to_track_grid,
};

const W: u32 = 320;
const H: u32 = 240;
const WIN: usize = 21;
const ITERS: usize = 30;
const LEVELS: usize = 4;

/// Deterministic band-limited texture (smoothed noise).
fn textured() -> GrayImage {
    let mut buf: Vec<f32> = (0..(W * H))
        .map(|i| {
            let mut s = i.wrapping_mul(2654435761) ^ 0x9e3779b9;
            s ^= s >> 15;
            s = s.wrapping_mul(0x85ebca6b);
            s ^= s >> 13;
            (s & 0xff) as f32
        })
        .collect();
    for _ in 0..3 {
        let src = buf.clone();
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let (mut sum, mut n) = (0.0, 0.0);
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && ny >= 0 && nx < W as i32 && ny < H as i32 {
                            sum += src[(ny as u32 * W + nx as u32) as usize];
                            n += 1.0;
                        }
                    }
                }
                buf[(y as u32 * W + x as u32) as usize] = sum / n;
            }
        }
    }
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            img.put_pixel(x, y, Luma([buf[(y * W + x) as usize] as u8]));
        }
    }
    img
}

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

fn main() {
    // Three-frame sequence with a known large inter-frame motion. A real system
    // gets this motion from the gyroscope; here we just know it.
    let base = textured();
    let motions = [(18.0f32, -12.0f32), (-15.0f32, 20.0f32)];
    let frames = [
        base.clone(),
        shift(&base, motions[0].0, motions[0].1),
        shift(
            &shift(&base, motions[0].0, motions[0].1),
            motions[1].0,
            motions[1].1,
        ),
    ];

    // Detect a uniform spread of corners in the first frame.
    let detected = good_features_to_track_grid(&frames[0], 6, 4, 3, 0.05, 12, &[]);
    let mut points: Vec<(f32, f32)> = detected
        .iter()
        .map(|&(x, y, _)| (x as f32, y as f32))
        .collect();
    println!("Frame 0: detected {} corners on a 6x4 grid", points.len());

    let mut ctx = TrackerContext::new();

    for (f, motion) in motions.iter().enumerate() {
        let prev = &frames[f];
        let next = &frames[f + 1];

        // Gyro-predicted feature motion (here, the known global shift).
        let predicted: Vec<(f32, f32)> = points
            .iter()
            .map(|&(x, y)| (x + motion.0, y + motion.1))
            .collect();

        ctx.prepare(prev, next, LEVELS);
        let results = ctx
            .track_fb(
                &points,
                Some(&predicted),
                WIN,
                ITERS,
                DEFAULT_MIN_EIGEN_THRESHOLD,
                DEFAULT_FB_THRESHOLD,
            )
            .to_vec();

        let mut counts = [0usize; 5];
        let mut next_points = Vec::new();
        for r in &results {
            counts[status_index(r.status)] += 1;
            if r.status == TrackStatus::Tracked {
                next_points.push(r.pos);
            }
        }

        println!(
            "\nFrame {} -> {} (predicted motion {:+.0},{:+.0}):",
            f,
            f + 1,
            motion.0,
            motion.1
        );
        println!(
            "  Tracked {}  OutOfBounds {}  Diverged {}  LowTexture {}  FbInconsistent {}",
            counts[0], counts[1], counts[2], counts[3], counts[4]
        );

        // Show a few per-point diagnostics.
        for (i, r) in results.iter().take(4).enumerate() {
            println!(
                "    pt{i}: ({:6.1},{:6.1}) {:?} error={:.3}",
                r.pos.0, r.pos.1, r.status, r.error
            );
        }

        points = next_points;
    }

    println!("\n{} points survived the full sequence", points.len());
}

fn status_index(s: TrackStatus) -> usize {
    match s {
        TrackStatus::Tracked => 0,
        TrackStatus::OutOfBounds => 1,
        TrackStatus::Diverged => 2,
        TrackStatus::LowTexture => 3,
        TrackStatus::FbInconsistent => 4,
    }
}
