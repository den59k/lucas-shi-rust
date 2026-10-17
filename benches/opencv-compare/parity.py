"""Native OpenCV parity and timing against the crate's dumped results.

Reads out/input1.pgm, out/input2.pgm and out/rust.json written by the native
crate, runs OpenCV with matched parameters, and writes out/native_results.json.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
WARMUP, REPS = 10, 100
MATCH_RADIUS = 1.5


def read_pgm(path):
    data = path.read_bytes()
    parts = data.split(maxsplit=4)
    assert parts[0] == b"P5" and parts[3] == b"255"
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(data[len(data) - w * h:], np.uint8).reshape(h, w)


def stats(samples):
    s = sorted(samples)
    pick = lambda q: s[round((len(s) - 1) * q)]
    return {"median": pick(0.5), "p90": pick(0.9), "min": s[0]}


def bench(fn):
    for _ in range(WARMUP):
        fn()
    t = []
    for _ in range(REPS):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1e3)
    return stats(t)


def nearest(a, b):
    """Fraction of rows of `a` within MATCH_RADIUS of some row of `b`, and the median distance."""
    if len(a) == 0 or len(b) == 0:
        return {"within": float("nan"), "median": float("nan")}
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return {"within": float((d <= MATCH_RADIUS).mean()), "median": float(np.median(d))}


def main():
    prev = read_pgm(OUT / "input1.pgm")
    nxt = read_pgm(OUT / "input2.pgm")
    rust = json.loads((OUT / "rust.json").read_text())
    p = rust["params"]
    results = {"opencv": cv2.__version__, "reps": REPS, "corners": {}, "tracking": {}, "timing": {}}

    rust_corners = np.array([[x, y] for x, y, _ in rust["corners"]], np.float64)
    for name, gradient in [("sobel3", 3), ("scharr", -1)]:
        try:
            c = cv2.goodFeaturesToTrack(prev, 0, p["quality"], p["min_distance"], mask=None, blockSize=3, gradientSize=gradient)
        except cv2.error as e:
            results["corners"][name] = {"error": str(e).strip().splitlines()[-1]}
            continue
        c = c.reshape(-1, 2).astype(np.float64)
        results["corners"][name] = {
            "opencv_count": int(len(c)),
            "crate_count": int(len(rust_corners)),
            "crate_within_radius_of_opencv": nearest(rust_corners, c),
            "opencv_within_radius_of_crate": nearest(c, rust_corners),
            "top100_overlap": nearest(rust_corners[:100], c[:100])["within"],
        }

    pts = np.array(rust["track_points"], np.float32).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, p["iterations"], 0.001)
    win = (p["window"], p["window"])
    lk = lambda: cv2.calcOpticalFlowPyrLK(prev, nxt, pts, None, winSize=win, maxLevel=p["levels"] - 1, criteria=crit)
    nxt_pts, st, _ = lk()
    ocv_ok = st.ravel() == 1
    rust_pos = np.array([[np.nan if t[0] is None else t[0], np.nan if t[1] is None else t[1]] for t in rust["track"]])
    rust_ok = np.array([t[2] == 0 for t in rust["track"]])
    both = ocv_ok & rust_ok
    d = np.linalg.norm(nxt_pts.reshape(-1, 2)[both] - rust_pos[both], axis=1)
    results["tracking"] = {
        "points": int(len(pts)),
        "opencv_tracked": int(ocv_ok.sum()),
        "crate_tracked": int(rust_ok.sum()),
        "status_agreement": float((ocv_ok == rust_ok).mean()),
        "both_tracked": int(both.sum()),
        "endpoint_dist_px": {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())} if both.any() else None,
    }

    gftt = lambda: cv2.goodFeaturesToTrack(prev, 0, p["quality"], p["min_distance"], blockSize=3)
    default_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    results["timing"]["OpenCV native, 1 thread"] = {"detect": bench(gftt), "track": bench(lk)}
    cv2.setNumThreads(default_threads)
    results["timing"][f"OpenCV native, {cv2.getNumThreads()} threads"] = {"detect": bench(gftt), "track": bench(lk)}
    results["timing"]["crate native"] = rust["native_ms"]

    (OUT / "native_results.json").write_text(json.dumps(results, indent=2))
    print(f"opencv {cv2.__version__}")
    print(json.dumps({k: v for k, v in results.items() if k not in ("timing", "opencv", "reps")}, indent=2))
    print("\n| Build | Detect median ms | Detect p90 | Track median ms | Track p90 |")
    print("|---|---:|---:|---:|---:|")
    for name, t in results["timing"].items():
        print(f"| {name} | {t['detect']['median']:.3f} | {t['detect']['p90']:.3f} | {t['track']['median']:.3f} | {t['track']['p90']:.3f} |")


if __name__ == "__main__":
    sys.exit(main())
