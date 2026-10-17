// Run the existing Criterion harness on Node's WASI preview1 runtime. The
// detector uses the repository's wasm32-wasip1 +simd128 target configuration.
const fs = require("node:fs");
const path = require("node:path");
const { WASI } = require("node:wasi");

async function main() {
    const [modulePath, ...args] = process.argv.slice(2);
    if (!modulePath) {
        throw new Error("usage: node benches/wasi-runner.cjs <features.wasm> [Criterion arguments]");
    }
    const reports = path.resolve(process.env.CRITERION_HOME || "target/criterion-wasi");
    fs.mkdirSync(reports, { recursive: true });
    const wasi = new WASI({
        version: "preview1",
        args: [modulePath, ...args],
        env: { CRITERION_HOME: "/reports" },
        preopens: { "/reports": reports },
        returnOnExit: true,
    });
    // Direct fd_write imports reproducibly crashed Node 22.22.2 during the
    // verbose baseline smoke check. A JS boundary avoids that host-call crash.
    // Detection and clock imports are unchanged; writes occur outside timing.
    const imports = wasi.getImportObject();
    const write = imports.wasi_snapshot_preview1.fd_write;
    imports.wasi_snapshot_preview1.fd_write = (...args) => write(...args);
    const { instance } = await WebAssembly.instantiate(
        fs.readFileSync(modulePath),
        imports,
    );
    process.exitCode = wasi.start(instance);
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
