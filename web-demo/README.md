# Optical-flow web demo

A tiny in-browser demo of the Lucas-Kanade tracker. Point your phone's camera at
a scene, **tap to drop a point**, and watch it ride the optical flow as the scene
(or the camera) moves. Everything runs client-side in WebAssembly — no server,
no upload.

```
web-demo/
├── index.html        # the whole UI (camera + canvas + tap handling)
├── src/lib.rs        # wasm-bindgen wrapper around TrackerContext
├── pkg/              # built wasm + JS glue (committed so it deploys as static files)
└── build.sh / .ps1   # rebuild pkg/ with wasm-pack
```

## Run locally

Camera access (`getUserMedia`) only works in a **secure context**: HTTPS, or
`http://localhost`. Any static file server over localhost works:

```bash
cd web-demo
python -m http.server 8080
# then open http://localhost:8080
```

On desktop you can test with your webcam. To try it on your phone over your LAN
you need HTTPS (a plain `http://192.168.x.x` will be blocked) — deploy it, or
use a tunneling tool that provides HTTPS (e.g. `cloudflared tunnel`, `ngrok`).

## Deploy (so you can open it on your phone)

`pkg/` is committed, so the folder is fully static — drop it on any HTTPS host:

- **GitHub Pages**: push the repo, enable Pages, point it at this folder (or copy
  `web-demo/` to the Pages root). Open the `https://…/web-demo/` URL on your phone.
- **Netlify / Vercel / Cloudflare Pages**: set the publish/output directory to
  `web-demo`. No build step is required because `pkg/` is prebuilt.

Then open the deployed URL on your phone, allow the camera, and tap the video.

## Rebuild the wasm

Only needed if you change `src/lib.rs` or the library:

```bash
cd web-demo
./build.sh          # or ./build.ps1 on Windows
```

This requires [`wasm-pack`](https://rustwasm.github.io/wasm-pack/). SIMD
(`+simd128`) is enabled automatically via the repo's `.cargo/config.toml`.

## How it works

Each animation frame the page draws the camera image into a small processing
canvas (longer side ≈ 360 px, chosen for real-time speed), converts it to
grayscale, and hands the bytes to `Tracker.push_frame`. The Rust side keeps the
previous frame, runs pyramidal Lucas-Kanade for every point, drops points that
leave the frame or fail to converge, and returns the survivors to be drawn.

Tracker parameters (pyramid levels, window size, iterations) are set in
[`src/lib.rs`](src/lib.rs) and tuned for phones; adjust there and rebuild.
