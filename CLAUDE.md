# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- **NEVER run git commands** — no `git add/commit/diff/log/status` or any git operation
- **Only `util/` files are readable and modifiable** — do NOT read, grep, or glob anything under `src/` or `include/`. The only code files available are `util/ssptycho.cpp` and `util/ptycho.cpp` (and other `util/*.cpp` as needed). Infer library behavior from how functions are called in `util/` code, never by reading their definitions
- **MUST cd to the workspace directory before running any executable** — the config uses relative `outputDir`, so running from the project directory writes output to the wrong place
- **Use `gpuerr()` only** — do NOT call `cudaDeviceSynchronize`
- **`ccmemMngr` = host (CPU) memory, `memMngr` = device (GPU) memory** — CPU code must NEVER directly access GPU pointers; always use `myMemcpyD2H`/`myMemcpyH2D` via host intermediary
- **`plt.*` functions only work on GPU images managed by the cuda image system** — never pass raw GPU pointers to plotting functions

## Build

```bash
# Set up environment (sets CDI_DIR, PATH, LD_LIBRARY_PATH, defines aliases)
source env.sh

# Incremental build
ninja -C build          # or: cdimake

# Full clean rebuild
ninja -C build clean && cmake -S . -B build --fresh && ninja -C build   # or: cdiremake

# Force rebuild a single util executable (e.g., ssptycho)
touch util/ssptycho.cpp && ninja -C build ssptycho_run
```

- Compiler: `clang++` (C++20), `nvcc` (CUDA)
- Build type: Release (`-O3 -DNDEBUG`)
- No subdirectory CMakeLists — single root `CMakeLists.txt` with auto-globbing
- `CMAKE_LINK_DEPENDS_NO_SHARED TRUE` (fast incremental rebuilds)

### Auto-Globbing Build Convention

The build system automatically converts source files to build targets:

| Source pattern | Build target | Example |
|---|---|---|
| `src/independant/*.c` | shared lib `<name>_c` | `imgio.c` → `libimgio_c.so` |
| `src/gpu/*.cu` | single shared lib `cudaLib` | all `.cu` → `libcudaLib.so` |
| `src/post_gpu/*.cc` | shared lib `<name>_cc` | `cdi.cc` → `libcdi_cc.so` |
| `util/*.cpp` | executable `<name>_run` | `cdi.cpp` → `cdi_run` |
| `util/gui/*.cpp` | GUI executable | `gui_cdi.cpp` → `gui_cdi_run` |

## Running Experiments

```bash
# ALWAYS cd to workspace first, then source env.sh, then run
cd /path/to/workspace
source /home/boyang/softwares/Imaging/CDIxx/env.sh
/home/boyang/softwares/Imaging/CDIxx/bin/<name>_run <config.cfg>

# Common executables:
#   cdi_run cdi.cfg           — single-shot CDI
#   ptycho_run ptycho.cfg     — scanning ptychography
#   ssptycho_run ptycho.cfg   — single-shot ptychography
#   ssBeam_run beam.cfg       — generate 9-spot pupil (writes image.bin for ssptycho)
#   takepsnr_run              — compute PSNR between images
```

Output goes to `outputDir/` (relative to CWD) as PNG files.

## Architecture

```
util/*.cpp  →  bin/*_run  (standalone executables, the only code you modify)
    │
src/post_gpu/*.cc  (C++ orchestration: CDI, ptycho, FISTA, propagation, plotting)
    │
src/gpu/*.cu  →  libcudaLib.so  (all CUDA kernels in one library)
    │
src/independant/*.c,*.cc  (CPU-only: memory pool, image I/O, config parser, LMDB, video)
    │
Hardware: CUDA/HIP, cuFFT, cuSOLVER, GSL, libpng, FFmpeg
```

### Memory Management

All allocations go through a custom pool. Use the convenience macros (not raw `borrowCache`):

```cpp
// GPU — declare + allocate:
myCuDMalloc(Real, d_data, numPixels);      // Real* d_data = (Real*)memMngr.borrowCache(...)
myCuDMallocClean(Real, d_data, numPixels); // zeroed

// GPU — allocate into existing pointer:
myCuMalloc(Real, d_data, numPixels);       // d_data = (Real*)memMngr.borrowCache(...)
myCuMallocClean(Real, d_data, numPixels);  // zeroed

// GPU — free:
myCuFree(d_data);  // memMngr.returnCache(d_data); d_data = 0

// CPU — declare + allocate:
myDMalloc(Real, h_data, numPixels);
myDMallocClean(Real, h_data, numPixels);

// CPU — allocate into existing pointer:
myMalloc(Real, h_data, numPixels);
myMallocClean(Real, h_data, numPixels);

// CPU — free:
myFree(h_data);
```

### CUDA Image System

GPU images for plotting must be registered with the image system before use:
```cpp
resize_cuda_image(cols, rows);  // MUST call before any plt.* or image operation
```

### Configuration

INI-style `.cfg` files parsed by `readConfig` (X-macro pattern in `readConfig.hpp`). Key fields:
- `runSim` — simulation vs real data
- `doIteration` — enable reconstruction
- `algorithm` — formula string like `"1000*HIO"` or `"100*HIO+100*POSER"`
- `outputDir` — where results are written (relative to CWD)
- `lambda`, `d`, `dpupil`, `pixelsize` — physical parameters
- `noiseLevel` — CCD noise (0 = ideal)

Example configs: `html/default/cdi.cfg`, `html/default/ptycho.cfg`, `html/default/pulse.cfg`

### Core Types

```cpp
#define Real          float
#define complexFormat _Complex Real
typedef unsigned short pixeltype;
enum Algorithm {RAAR, ER, POSER, HIO, POSHIO, FHIO, shrinkWrap, XCORRELATION, KKT, TV, cnt};
```

## ssptycho (Single-Shot Ptychography) — Key Patterns

- Reads `image.bin` from `ssBeam_run` (9-spot pupil, complex, 2048×2048)
- Propagates pupil via `angularSpectrumPropagate` (distance `dpupil`) to sample plane
- `readComplexWaveFront` reads 1024×1024 PNGs, overwrites `row_O`/`column_O` to PNG size, allocates GPU data directly (NOT via memMngr)
- Must manually call `pad()` to expand 1024×1024 → 2048×2048, then `resize_cuda_image(2048, 2048)`, `allocateMem()`, `createWaveFront()`
- Do NOT call `memMngr.returnCache()` on buffers from `readComplexWaveFront` (allocated directly, not via memMngr)
- `readImage()` returns host pointer — must allocate device buffer via `myCuMalloc` before GPU use
- Pattern plotted with `plotFloat(..., MOD, 0, 1, "name", 1, 0, 1)` for log+color
- Spatial wavefronts use `plotComplexColor`, detector patterns use log+color

## Testing

No unit test framework. Testing is done by:
1. Running the binary in `bin/` with a config file
2. Comparing output PNGs against reference images
3. Using `takepsnr_run` to compute PSNR between reference and reconstruction

## HIP/AMD GPU Support

```bash
cmake -S . -B build -G Ninja -DUSE_HIP=ON
ninja -C build
```
Auto-converts `.cu` → `.hip` via `hipify-clang`, replaces `cub::` → `hipcub::`, links HIP libraries.

## Detailed Documentation

See `README.md` for complete file-by-file documentation of all headers, libraries, executables, config parameters, and data flow diagrams. See `AGENTS.md` for additional repository guidelines.
