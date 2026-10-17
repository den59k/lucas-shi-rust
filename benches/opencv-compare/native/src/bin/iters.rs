use std::path::Path;
use std::time::Instant;
use image::GrayImage;
use optical_flow_lk::{DEFAULT_MIN_EIGEN_THRESHOLD, TrackStatus, TrackerContext, good_features_to_track};

fn median(mut v: Vec<f64>) -> f64 { v.sort_by(f64::total_cmp); v[v.len() / 2] }
fn time<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..10 { f(); }
    median((0..60).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() * 1e3 }).collect())
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let prev: GrayImage = image::open(root.join("examples/input1.png")).unwrap().into_luma8();
    let next: GrayImage = image::open(root.join("examples/input2.png")).unwrap().into_luma8();
    let corners = good_features_to_track(&prev, 0.01, 5);
    let points: Vec<(f32, f32)> = corners.iter().take(200).map(|&(x, y, _)| (x as f32, y as f32)).collect();
    let mut ctx = TrackerContext::new();
    println!("prepare (both pyramids, 4 levels): {:.3} ms", time(|| ctx.prepare(&prev, &next, 4)));
    ctx.prepare(&prev, &next, 4);
    for iters in [1usize, 3, 5, 10, 20, 30] {
        let ms = time(|| { std::hint::black_box(ctx.track(&points, None, 21, iters, DEFAULT_MIN_EIGEN_THRESHOLD)); });
        let r = ctx.track(&points, None, 21, iters, DEFAULT_MIN_EIGEN_THRESHOLD);
        let count = |s: TrackStatus| r.iter().filter(|x| x.status == s).count();
        println!("track only, max_iterations={iters:2}: {ms:7.3} ms  tracked={} diverged={} lowtex={} oob={}",
            count(TrackStatus::Tracked), count(TrackStatus::Diverged), count(TrackStatus::LowTexture), count(TrackStatus::OutOfBounds));
    }
    for win in [7usize, 11, 15, 21] {
        let ms = time(|| { std::hint::black_box(ctx.track(&points, None, win, 30, DEFAULT_MIN_EIGEN_THRESHOLD)); });
        println!("track only, window={win:2}, 30 iters: {ms:7.3} ms");
    }
    for thr in [1e-3f32, 1e-4, 1e-5, 0.0] {
        let r = ctx.track(&points, None, 21, 30, thr);
        println!("min_eigen_threshold={thr:e}: tracked={} lowtex={} diverged={}",
            r.iter().filter(|x| x.status == TrackStatus::Tracked).count(),
            r.iter().filter(|x| x.status == TrackStatus::LowTexture).count(),
            r.iter().filter(|x| x.status == TrackStatus::Diverged).count());
    }
}
