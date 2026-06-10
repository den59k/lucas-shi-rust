# Lucas Canade Optical Flow and Shi-Tomasi feature detection on Rust

[![Crates.io](https://img.shields.io/crates/v/optical-flow-lk)](https://crates.io/crates/optical-flow-lk)
[![Documentation](https://docs.rs/optical-flow-lk/badge.svg)](https://docs.rs/optical-flow-lk)

High-performance Rust implementation of Lucas-Kanade optical flow and Shi-Tomasi feature detection, optimized for real-time applications and WebAssembly (Wasm) compatibility.

## Features

- 🎯 Pyramidal Lucas-Kanade optical flow with per-point status and photometric error
- 🔁 Forward-backward consistency check to reject occlusions and outliers
- 🧭 Optional motion prediction (initial guess) for large inter-frame displacements
- 🔍 Shi-Tomasi feature detection, plus grid-based detection for uniform coverage
- ♻️ Zero-allocation steady-state path (`TrackerContext`) for real-time per-frame tracking
- ⚡ SIMD-accelerated gradients and pyramid: AVX2 (x86), NEON (aarch64), `simd128` (wasm)
- 🌐 Built on the [`image`](https://crates.io/crates/image) crate; WebAssembly-ready

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
optical-flow-lk = "0.3"
```

Basic example — detect corners in one frame and track them into the next:
```rust
use image::{open, GrayImage};
use optical_flow_lk::{
    build_pyramid, calc_optical_flow_ex, good_features_to_track,
    TrackStatus, DEFAULT_MIN_EIGEN_THRESHOLD,
};

let prev: GrayImage = open("examples/input1.png").unwrap().into_luma8();
let next: GrayImage = open("examples/input2.png").unwrap().into_luma8();

let prev_pyr = build_pyramid(&prev, 4);
let next_pyr = build_pyramid(&next, 4);

let mut corners = good_features_to_track(&prev, 0.1, 5);
corners.truncate(100);
let points: Vec<(f32, f32)> = corners.iter().map(|&(x, y, _)| (x as f32, y as f32)).collect();

// `None` = no initial guess; 21px window, 30 iterations per level.
let results = calc_optical_flow_ex(
    &prev_pyr, &next_pyr, &points, None, 21, 30, DEFAULT_MIN_EIGEN_THRESHOLD,
);

for (start, r) in points.iter().zip(&results) {
    if r.status == TrackStatus::Tracked {
        println!("{start:?} -> {:?} (error {:.1})", r.pos, r.error);
    }
}
```

### Real-time tracking

For per-frame tracking (e.g. a VIO front-end or the web demo), reuse a
[`TrackerContext`]. After the first frame it performs no heap allocation, and
`track_fb` adds the forward-backward consistency check:

```rust
use optical_flow_lk::{TrackerContext, DEFAULT_MIN_EIGEN_THRESHOLD, DEFAULT_FB_THRESHOLD};

let mut ctx = TrackerContext::new();

// Per frame pair (`prev`, `next` are `&GrayImage`):
ctx.prepare(&prev, &next, 4);
let results = ctx.track_fb(
    &points, None, 21, 30, DEFAULT_MIN_EIGEN_THRESHOLD, DEFAULT_FB_THRESHOLD,
);
// Points flagged `TrackStatus::FbInconsistent` failed the round-trip.
```

[`TrackerContext`]: https://docs.rs/optical-flow-lk/latest/optical_flow_lk/struct.TrackerContext.html

> The original `calc_optical_flow` is still available but **deprecated** since
> 0.3.0 — prefer `calc_optical_flow_ex` (status + error) or `TrackerContext`.

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
