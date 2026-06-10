#!/usr/bin/env bash
# Rebuild the wasm package. Run from the web-demo/ directory.
# Requires: wasm-pack (https://rustwasm.github.io/wasm-pack/).
# The parent .cargo/config.toml enables +simd128 automatically.
set -euo pipefail
wasm-pack build --target web --release --out-dir pkg
rm -f pkg/.gitignore
echo "Built pkg/. Serve this folder over HTTPS (or localhost) to run the demo."
