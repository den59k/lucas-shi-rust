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
