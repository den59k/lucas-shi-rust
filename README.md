# Lucas Canade Optical Flow and Shi-Tomasi feature detection on Rust

[![Crates.io](https://img.shields.io/crates/v/optical-flow-lk)](https://crates.io/crates/optical-flow-lk)
[![Documentation](https://docs.rs/optical-flow-lk/badge.svg)](https://docs.rs/optical-flow-lk)

High-performance Rust implementation of Lucas-Kanade optical flow and Shi-Tomasi feature detection, optimized for real-time applications and WebAssembly (Wasm) compatibility.

## Features

- 🔍 Efficient feature point detection using Shi-Tomasi
- 🖼️ Integration with `image` and `imageproc` crates
- 🌐 WebAssembly (Wasm) compatible

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
optical-flow-lk = "0.1"
```

Basic example:
```rust
use image::{open, GrayImage, Rgba};
use optical_flow_lk::{build_pyramid, calc_optical_flow, good_features_to_track};

let prev_frame: GrayImage = open("examples/input1.png").unwrap().clone().into_luma8();
let next_frame: GrayImage = open("examples/input2.png").unwrap().clone().into_luma8();

let prev_frame_pyr = build_pyramid(&prev_frame, 4);
let next_frame_pyr = build_pyramid(&next_frame, 4);

let mut points = good_features_to_track(&prev_frame, 0.1, 5);
points.truncate(100);
let prev_points: Vec<(f32, f32)> = points.iter().map(|&x| (x.0 as f32, x.1 as f32)).collect();

let next_points = calc_optical_flow(&prev_frame_pyr, &next_frame_pyr, &prev_points, 21, 30);
```

## Live demo

A browser demo runs the tracker entirely client-side in WebAssembly: point your
phone's camera at a scene and tap to drop points (or hit **Auto** to detect
Shi-Tomasi corners) and watch them ride the optical flow.

**▶ [lk-demo.jt3.ru](https://lk-demo.jt3.ru)**

Source and build instructions are in [`web-demo/`](web-demo/).

## WebAssembly

The Scharr gradient kernel has a hand-written `simd128` path that is selected
automatically on `wasm32` — **but only when the target is built with SIMD
enabled**, since WASM has no runtime feature detection. The bundled
[`.cargo/config.toml`](.cargo/config.toml) sets this for you:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]
```

If you build from a different working directory (so that config is not picked
up), pass it yourself:

```bash
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
```

Without `+simd128` the crate still works, falling back to a scalar gradient
loop. For production WASM, also run `wasm-opt -O3` on the output. On a 640×480
per-frame track step, the simd128 build plus the bounds-check-free bilinear
sampler is roughly **2× faster** than the scalar build in V8 (Node).
