# Lucas Canade Optical Flow and Shi-Tomasi feature detection on Rust

[![Crates.io](https://img.shields.io/crates/v/optical-flow-lk)](https://crates.io/crates/optical-flow-lk)
[![Documentation](https://docs.rs/optical-flow-lk/badge.svg)](https://docs.rs/optical-flow-lk)

High-performance Rust implementation of Lucas-Kanade optical flow and Shi-Tomasi feature detection, optimized for real-time applications and WebAssembly (Wasm) compatibility.

## Features

- 🎯 Pyramidal Lucas-Kanade optical flow with per-point status and photometric error
- 🔁 Forward-backward consistency check to reject occlusions and outliers
- 🧭 Optional motion prediction (initial guess) for large inter-frame displacements
- 🔍 Shi-Tomasi feature detection, plus grid-based detection for uniform coverage
- ♻️ Zero-allocation steady-state path (`TrackerContext`, `DetectorContext`) for real-time per-frame tracking, with up-front capacity reservation
- 🌡️ f32 pipeline for images in physical units, e.g. thermal frames in °C, without 8-bit quantization
- ⚡ SIMD-accelerated gradients and pyramid: AVX2 (x86), NEON (aarch64), `simd128` (wasm)
- 🌐 Built on the [`image`](https://crates.io/crates/image) crate; WebAssembly-ready

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
optical-flow-lk = "0.5"
```

Basic example — detect corners in one frame and track them into the next:
```rust
use image::{open, GrayImage};
use optical_flow_lk::{
    build_pyramid, calc_optical_flow_ex, good_features_to_track,
    Corner, TrackParams, TrackStatus, DEFAULT_MIN_EIGEN_THRESHOLD,
};

let prev: GrayImage = open("examples/input1.png").unwrap().into_luma8();
let next: GrayImage = open("examples/input2.png").unwrap().into_luma8();

let prev_pyr = build_pyramid(&prev, 4);
let next_pyr = build_pyramid(&next, 4);

// Keep corners scoring at least 1% of the strongest one, at least 5px apart.
let mut corners = good_features_to_track(&prev, 0.01, 5);
corners.truncate(100);
let points: Vec<(f32, f32)> = corners.iter().map(Corner::position).collect();

// 21px window, 30 iterations per level; `None` = no initial guess.
let params = TrackParams {
    window_size: 21,
    max_iterations: 30,
    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
};
let results = calc_optical_flow_ex(&prev_pyr, &next_pyr, &points, None, params);

for (start, r) in points.iter().zip(&results) {
    if r.status == TrackStatus::Tracked {
        println!("{start:?} -> {:?} (error {:.1})", r.pos, r.error);
    }
}
```

### Real-time tracking

For per-frame tracking (e.g. a VIO front-end or the web demo), reuse a
[`TrackerContext`] and pass it each new frame with `advance`. The pyramid of
the last frame becomes the previous one, so only the new frame's pyramid is
built and you keep no copy of the last frame. After the first frame it
performs no heap allocation, and `track_fb` adds the forward-backward
consistency check:

```rust
use optical_flow_lk::{TrackerContext, DEFAULT_FB_THRESHOLD};

let mut ctx = TrackerContext::new();

// Per frame (`frame` is a `&GrayImage`), with `params` as above:
ctx.advance(frame, 4);
let results = ctx.track_fb(&points, None, params, DEFAULT_FB_THRESHOLD);
// Points flagged `TrackStatus::FbInconsistent` failed the round-trip.
```

`advance(frame, 4)` gives the same results, bit for bit, as
`prepare(&last_frame, frame, 4)`, which builds both pyramids from an explicit
frame pair. On the first call there is no last frame yet, so both pyramids
show `frame` and tracking measures no motion. `FloatTrackerContext` has the
same `advance`.

Detection has the same shape. A [`DetectorContext`] owns the gradient, tensor
and score planes plus the spacing grid, so `detect` / `detect_grid` allocate
nothing after the first frame:

```rust
use optical_flow_lk::{DetectorContext, Grid};

let mut detector = DetectorContext::new();
// Top up lost tracks every few frames: 6x4 grid, 3 per cell, quality 0.05,
// at least 12 px from each other and from the points already tracked.
let grid = Grid { cols: 6, rows: 4, max_per_cell: 3 };
let fresh = detector.detect_grid(&prev, grid, 0.05, 12, &points);
```

To keep even the first frame free of allocation, create the contexts with
`with_capacity`. A `DetectorCapacity` reserves for the largest possible
candidate count, so no image content can grow a buffer later:

```rust
use optical_flow_lk::{DetectorCapacity, DetectorContext, TrackerCapacity, TrackerContext};

let mut detector = DetectorContext::with_capacity(DetectorCapacity {
    width: 640, height: 480, min_distance: 12, grid_cells: 24, existing_points: 200,
});
let mut tracker = TrackerContext::with_capacity(TrackerCapacity {
    width: 640, height: 480, levels: 4, window_size: 21, points: 200,
});
```

### f32 images in physical units

For sensors whose values carry a unit, such as thermal cameras in °C,
[`FloatDetectorContext`] and [`FloatTrackerContext`] run the same algorithms on
f32 frames, without range normalization or 8-bit quantization. Scores and the
tracker's eigenvalue threshold are in (unit / pixel)^2, residuals in the unit
itself. A [`FloatImageView`] borrows a frame after checking that every value is
finite and within range:

```rust
use optical_flow_lk::{
    Corner, FloatDetectorContext, FloatImageView, FloatTrackerContext, ScoreThreshold,
    TrackParams, DEFAULT_FB_THRESHOLD,
};

// `prev_celsius` and `next_celsius` are 640x480 row-major `&[f32]` frames.
let prev = FloatImageView::new(prev_celsius, 640, 480)?;
let next = FloatImageView::new(next_celsius, 640, 480)?;

// Corners scoring at least 0.002 (°C / px)^2, at least 8 px apart.
let mut detector = FloatDetectorContext::new();
let corners = detector.detect(prev, ScoreThreshold::absolute(0.002), 8);
let points: Vec<(f32, f32)> = corners.iter().map(Corner::position).collect();

// The eigenvalue threshold is in (°C / px)^2 as well and has no default.
let params = TrackParams { window_size: 15, max_iterations: 15, min_eigen_threshold: 1e-4 };
let mut tracker = FloatTrackerContext::new();
tracker.prepare(prev, next, 3);
let results = tracker.track_fb(&points, None, params, DEFAULT_FB_THRESHOLD);
```

On frames holding 8-bit values the f32 detector returns the 8-bit detector's
corners bit for bit. On a 640x480 frame on an Apple M2, a whole frame step
(pyramids, forward-backward tracking, grid top-up) costs about 1.1x the 8-bit
pipeline on the same data, natively and under WASI; tracking alone is faster in
f32. See [the measurements](benches/results/float-vs-u8-2026-10-17.md).
Against the published 0.3.1, the f32 frame step is 3.3-3.6x faster natively
and 2.9-3.1x faster under WASI ([comparison](benches/results/v0.3.1-vs-current-2026-10-17.md)).

[`TrackerContext`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.TrackerContext.html
[`DetectorContext`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.DetectorContext.html
[`FloatDetectorContext`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.FloatDetectorContext.html
[`FloatTrackerContext`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.FloatTrackerContext.html
[`FloatImageView`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.FloatImageView.html

> Upgrading from 0.4: detection returns [`Corner`]s, grid detection takes a
> [`Grid`], and tracking takes [`TrackParams`]. The deprecated
> `calc_optical_flow` is gone; use `calc_optical_flow_ex` or `TrackerContext`.
> See the [changelog](CHANGELOG.md).

[`Corner`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.Corner.html
[`Grid`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.Grid.html
[`TrackParams`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.TrackParams.html

## Performance

Measured against OpenCV 4.11 on the bundled 720×405 frame pair at 0.5.0 on
2026-10-20. The harness, parameters and caveats are in
[`benches/opencv-compare/`](benches/opencv-compare/).

On a phone, the deployment target: Xiaomi 11T (MediaTek Dimensity 1200,
arm64), Android 14, Chrome 155. Medians per call over 100 samples of 5 calls,
steady-state rounds after warm-up.

| | Detect, ms | Track 200 points, ms |
|---|---:|---:|
| this crate, wasm `simd128` | 2.9 | 14.7 |
| OpenCV.js 4.11 (wasm, scalar build) | 9.9 | 16.1 |

On a desktop, AMD Ryzen 9 9900X, wasm under Node 24. Medians of 100 calls.

| | Detect, ms | Track 200 points, ms |
|---|---:|---:|
| this crate, wasm `simd128` | 1.1 | 4.2 |
| OpenCV.js 4.11 (wasm, scalar build) | 2.5 | 5.8 |
| this crate, native (AVX2) | 0.7 | 3.4 |
| OpenCV native, 1 thread | 2.0 | 2.3 |

The desktop wasm detection figure was 0.8 ms at 0.4.1. The grouped
non-maximum suppression of 0.5.0 is faster natively, on the phone and on the
synthetic benchmark inputs, but slower on this frame under V8 on x86-64.
Details and the per-input figures are in the comparison README.

Corners agree with OpenCV's to the pixel: every corner this crate finds has an
OpenCV Scharr corner within 1.5 px, and 99% of OpenCV's have one of this
crate's. Tracked endpoints agree to a median of 0.004 px on points both
libraries track.

## Live demo

A browser demo runs the tracker entirely client-side in WebAssembly: point your
phone's camera at a scene and tap to drop points (or hit **Auto** to detect
Shi-Tomasi corners) and watch them ride the optical flow.

**▶ [lk-demo.jt3.ru](https://lk-demo.jt3.ru)**

Source and build instructions are in [`web-demo/`](web-demo/).

## WebAssembly

The hot image kernels — the Scharr gradients and the pyramid downsample — have
hand-written `simd128` paths that are selected automatically on `wasm32`
— **but only when the target is built with SIMD enabled**, since WASM has no
runtime feature detection. The bundled [`.cargo/config.toml`](.cargo/config.toml)
sets this for you:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]
```

If you build from a different working directory (so that config is not picked
up), pass it yourself:

```bash
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
```

Without `+simd128` the crate still works, falling back to scalar loops. For
production WASM, also run `wasm-opt -O3` on the output (`wasm-pack` does this
automatically). On a 640×480 per-frame track step, the `simd128` build plus the
bounds-check-free bilinear sampler is roughly **2× faster** than the scalar
build in V8 (Node).
