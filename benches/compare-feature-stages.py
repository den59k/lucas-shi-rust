#!/usr/bin/env python3
"""Compare staged detection work without modifying either library revision.

Usage: python3 benches/compare-feature-stages.py [before-commit] [output-directory]
Requires installed stable wasm32-wasip1 and Node's WASI preview1 runtime.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile


REPO = Path(__file__).resolve().parents[1]
BEFORE = sys.argv[1] if len(sys.argv) > 1 else "0ff69b9"
ROOT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(tempfile.mkdtemp(prefix="lk-stages-"))
ROOT.mkdir(parents=True, exist_ok=True)
print(f"Run directory: {ROOT}", flush=True)


def command(args, cwd=REPO, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, stderr=subprocess.STDOUT).decode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = ROOT / "before-source"
before.mkdir()
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git", "archive", BEFORE], cwd=REPO))) as archive:
    archive.extractall(before)  # Trusted local git archive, not external input.
for name in ["Cargo.toml", "Cargo.lock"]:
    shutil.copy2(REPO / name, before / name)
shutil.copytree(REPO / "benches", before / "benches", dirs_exist_ok=True)
# Adapt only the benchmark wrapper to historical private function names. The
# baseline src/ tree stays byte-identical to git archive BEFORE.
access = before / "benches/support/feature_stage_access.rs"
access.write_text(access.read_text().replace("box_sum_3x3_in_place", "box_filter_3x3_in_place")
                  .replace("min_eigenvalue_scores", "compute_min_eigenvalues"))

metadata = {
    "before_commit": command(["git", "rev-parse", BEFORE]).strip(),
    "after_commit": command(["git", "rev-parse", "HEAD"]).strip(),
    "after_diff": command(["git", "diff", "--", "src", "Cargo.toml", "Cargo.lock"]),
    "rustc": command(["rustc", "+stable", "-vV"]).strip(),
    "node": command(["node", "--version"]).strip(),
    "host": platform.platform(),
    "criterion": "0.8.2", "samples": 30, "warmup_seconds": 0.3,
    "measurement_seconds": 1, "bootstrap_resamples": 10000,
    "root": str(ROOT), "hashes": {},
    "order": "all builds and smoke checks first, native then WASI; before then after per input",
    "baseline_overlay": "current manifest, lockfile, benchmarks; only benchmark wrapper identifiers adapted",
}
for phase, source in [("before", before), ("after", REPO)]:
    metadata["hashes"][phase] = {
        str(p.relative_to(source)): sha(p)
        for p in [source / "Cargo.toml", source / "Cargo.lock", source / "src/features.rs",
                  source / "src/utils/box_filter_3x3.rs", source / "src/utils/fast_gradients.rs",
                  source / "benches/feature_stages.rs", source / "benches/support/feature_stage_access.rs",
                  source / "benches/support/feature_inputs.rs"]
    }
(ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

binaries = {}
for phase, source in [("before", before), ("after", REPO)]:
    for runtime in ["native", "wasi"]:
        args = ["cargo", "+stable", "bench", "--locked", "--offline", "--bench", "feature_stages", "--no-run", "--message-format=json"]
        env = os.environ.copy()
        if phase == "before":
            env["CARGO_TARGET_DIR"] = str(ROOT / "before-target")
        if runtime == "wasi":
            args.extend(["--target", "wasm32-wasip1"])
        print(f"Building {runtime}/{phase}", flush=True)
        output = command(args, cwd=source, env=env)
        (ROOT / f"build-{runtime}-{phase}.log").write_text(output)
        artifacts = []
        for line in output.splitlines():
            if line.startswith("{"):
                item = json.loads(line)
                if item.get("reason") == "compiler-artifact" and item["target"]["name"] == "feature_stages" and item.get("executable"):
                    artifacts.append(Path(item["executable"]))
        assert len(artifacts) == 1, artifacts
        dest = ROOT / f"{runtime}-{phase}{'.wasm' if runtime == 'wasi' else ''}"
        shutil.copy2(artifacts[0], dest)
        binaries[runtime, phase] = dest
        metadata["hashes"][phase][runtime + "_binary"] = sha(dest)


def run(runtime, phase, args, label):
    executable = binaries[runtime, phase]
    cmd = (["node", str(REPO / "benches/wasi-runner.cjs"), str(executable)]
           if runtime == "wasi" else [str(executable)]) + args
    env = os.environ.copy()
    env["CRITERION_HOME"] = str(ROOT / "reports" / runtime / phase)
    with (ROOT / f"{runtime}-{phase}-{label}.log").open("w") as log:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


# Assertions verify included-source/public-library parity, threshold equivalence,
# and deterministic fixtures before starting any timed runs.
for runtime in ["native", "wasi"]:
    for phase in ["before", "after"]:
        print(f"Smoke check {runtime}/{phase}", flush=True)
        run(runtime, phase, ["--test", "--noplot"], "smoke")

for runtime in ["native", "wasi"]:
    for case in ["example_720x405", "texture_320x240", "texture_640x480", "low_contrast_320x240", "flat_640x480", "controlled"]:
        for phase in ["before", "after"]:
            print(f"Measuring {runtime}/{case}/{phase}", flush=True)
            run(runtime, phase, [f"feature_stages/{case}/", "--bench", "--noplot"], case)

records = []
for runtime in ["native", "wasi"]:
    for phase in ["before", "after"]:
        report = ROOT / "reports" / runtime / phase / "feature_stages"
        for estimate in sorted(report.glob("**/new/estimates.json")):
            parts = estimate.relative_to(report).parts
            # Criterion sanitizes '/' in a function-only ID to '_'; structured
            # BenchmarkId(case, stage) uses two directory levels instead.
            case, stage = (("controlled", parts[0][len("controlled_"):])
                           if parts[0].startswith("controlled_") else parts[:2])
            records.append({"runtime": runtime, "phase": phase, "case": case, "stage": stage,
                            "estimates_ns": json.loads(estimate.read_text())})
        for line in (ROOT / f"{runtime}-{phase}-smoke.log").read_text().splitlines():
            for tag in ["COUNTS", "FLOOR", "CONTROL"]:
                if line.startswith(tag + " "):
                    records.append({"runtime": runtime, "phase": phase, "kind": tag.lower(), **json.loads(line[len(tag) + 1:])})
controls = [{k: v for k, v in r.items() if k not in ["runtime", "phase"]}
            for r in records if r.get("kind") == "control"]
assert len(controls) == 4 and all(c == controls[0] for c in controls)
for phase in ["before", "after"]:
    def counts(runtime):
        return [{k: v for k, v in r.items() if k != "runtime"} for r in records
                if r.get("kind") == "counts" and r["runtime"] == runtime and r["phase"] == phase]
    assert len(counts("native")) == 5 and counts("native") == counts("wasi")
assert len([r for r in records if "estimates_ns" in r]) == 256
metadata["validation"] = "all four control fixtures identical; native/WASI counts equal per revision; 256 timing records"
(ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(ROOT / "results.json").write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n")
print(f"Finished: {ROOT / 'results.json'}", flush=True)
