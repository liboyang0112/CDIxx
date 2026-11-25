# Repository Guidelines

CDIxx is a C++/CUDA toolkit for coherent diffraction imaging (CDI), ptychography, and related phase-reconstruction algorithms.

## Project Structure

- `util/*.cpp` — standalone entry points; each builds to an executable `<name>_run` in `bin/` (e.g., `cdi.cpp` → `cdi_run`, `ssptycho.cpp` → `ssptycho_run`). This is the layer to modify for new features.
- `src/post_gpu/*.cc` — C++ orchestration: algorithms (CDI, ptychography, FISTA), propagation, plotting.
- `src/gpu/*.cu` — all CUDA kernels, compiled into a single `libcudaLib.so`.
- `src/independant/*.{c,cc}` — CPU-only: memory pool, image I/O, config parser, LMDB, video.
- `include/` — shared headers (config parser uses an X-macro pattern in `readConfig.hpp`).
- `python/` — auxiliary scripts (training, data I/O); `data/` — element refraction data; `html/` — web UI and default configs (e.g., `html/default/cdi.cfg`).
- `build/`, `bin/`, `lib/` — build artifacts (git-ignored); `compile_commands.json` symlinks into `build/`.

## Build & Run

```bash
source env.sh                    # sets CDI_DIR, PATH, LD_LIBRARY_PATH; defines cdimake/cdiremake
ninja -C build                   # incremental build (alias: cdimake)
ninja -C build clean && cmake -S . -B build --fresh && ninja -C build   # full rebuild (alias: cdiremake)
touch util/ssptycho.cpp && ninja -C build ssptycho_run                  # rebuild one target
```

- Compiler: `clang++` (C++20) + `nvcc`; Release build (`-O3 -DNDEBUG`). Single root `CMakeLists.txt` auto-globs sources.
- AMD GPU: `cmake -S . -B build -DUSE_HIP=ON` (auto-hipifies `.cu` → `.hip`).
- **Always `cd` into the experiment workspace before running a binary** — configs use relative `outputDir`, so outputs land in `outputDir/` relative to CWD:

```bash
cd /path/to/workspace && source $CDI_DIR/env.sh
$CDI_DIR/bin/cdi_run cdi.cfg        # or ptycho_run, ssptycho_run, ssBeam_run, takepsnr_run
```

## Coding Style

- Use the memory-pool macros, never raw `borrowCache`/`malloc`: GPU `myCuDMalloc`/`myCuMalloc`/`myCuFree`, CPU `myDMalloc`/`myMalloc`/`myFree` (add `Clean` variants for zeroed buffers).
- `ccmemMngr` is host memory, `memMngr` is device memory; use `myMemcpyD2H`/`myMemcpyH2D` for transfers — never dereference GPU pointers from CPU code.
- Use `gpuerr()` for device sync/error checks; do not call `cudaDeviceSynchronize` directly.
- GPU images for plotting must be registered first: `resize_cuda_image(cols, rows)` before any `plt.*` call.
- Type aliases: `Real` (float), `complexFormat`; algorithms are an enum (`HIO`, `POSER`, `RAAR`, ...), configured via strings like `"1000*HIO"`.

## Testing

There is no unit-test framework. Verify changes by:

1. Building the affected `<name>_run` target.
2. Running it against a config (`html/default/*.cfg` are good starting points) from a scratch workspace.
3. Comparing output PNGs to references, and computing similarity with `takepsnr_run`.

## Commits & PRs

Commit messages are short, imperative, and lowercase (e.g., `fix ssptycho web`, `add correction include`, `optim writePNG`). No enforced format. For changes, describe the affected executable/configs and include example outputs or PSNR comparisons in the PR description; link related issues when applicable.
