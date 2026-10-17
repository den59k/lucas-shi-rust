#!/usr/bin/env bash
# Builds and runs the whole OpenCV comparison. Run from this directory.
set -euo pipefail
cd "$(dirname "$0")"

echo "== native crate: dump inputs, crate results, native timings =="
(cd native && cargo build --release --quiet)
native/target/release/opencv-compare-native out

echo "== wasm wrapper: simd128 build (parent .cargo/config.toml enables +simd128) =="
(cd wasm && wasm-pack build --target nodejs --release --out-dir pkg-simd)
echo "== wasm wrapper: scalar build =="
(cd wasm && RUSTFLAGS="-C target-feature=-simd128" wasm-pack build --target nodejs --release --out-dir pkg-scalar -- --target-dir target-scalar)

echo "== OpenCV.js =="
npm install --silent
node harness.cjs

echo "== native OpenCV =="
python -I parity.py
