# CDIxx CUDA Operation API Reference

This document catalogs the CUDA / GPU operation APIs used across CDIxx (the
coherent diffraction imaging toolkit) with concrete usage examples taken from
the codebase. Every example below is a real, working call.

All GPU work goes through the memory-pool managers and the helper wrappers in
`include/cudaConfig.hpp`, `include/memManager.hpp`, `include/propagator.hpp`
and `include/cuPlotter.hpp`. Kernels live in `src/gpu/*.cu` and are compiled
into a single `libcudaLib.so`.

## 1. Core types (`include/format.hpp`)

- `Real` — `float` alias. All GPU scalar data is single precision.
- `complexFormat` — `float2` alias (`{float x, float y}` = real/imag). The
  workhorse type for wavefronts / fields.
- `pixeltype` — `unsigned short` (16-bit) when `CCDBits==16`, else `uchar`.
  Used as the grayscale plot cache type.
- `col_rgb` — `struct { char r, g, b; }`. Used as the color plot cache type.
- Complex helpers (usable host & device): `crealf`, `cimagf`, `cabs`, `carg`,
  `conj`, plus `float2` arithmetic operators (`+`, `-`, `*`, `/`, `+=`, ...).
- `sq(x)` — `x*x` macro.

Example (building a field on the host then uploading):
```cpp
myDMalloc(complexFormat, h_pupil, N*N);
for(int i = 0; i < N; i++)
  for(int j = 0; j < N; j++){
    float x = j - N/2, y = i - N/2;
    h_pupil[i*N+j] = complexFormat(expf(-(x*x+y*y)/(w0*w0)), 0);
  }
myMemcpyH2D(d_pupil, h_pupil, N*N*sizeof(complexFormat));
myFree(h_pupil);
```
(`util/obliqueDemo.cpp`)

## 2. Memory management (`memManager.hpp`, `cudaConfig.hpp`)

Two pools: `memMngr` (device / GPU) and `ccmemMngr` (host / CPU). Always use
the macros, never raw `borrowCache`/`malloc`.

### GPU (device) memory
| macro | zeroed? | declares pointer? |
|-------|---------|-------------------|
| `myCuMalloc(T, var, n)`   | no | no (`var` pre-declared) |
| `myCuMallocClean(T, var, n)` | yes | no |
| `myCuDMalloc(T, var, n)`  | no | yes (declares `T* var`) |
| `myCuDMallocClean(T, var, n)` | yes | yes |
| `myCuFree(ptr)`           | —   | frees & nulls `ptr` |

### CPU (host) memory
| macro | zeroed? | declares pointer? |
|-------|---------|-------------------|
| `myMalloc(T, var, n)` / `myDMalloc(T, var, n)` | no | yes |
| `myMallocClean(T, var, n)` / `myDMallocClean(T, var, n)` | yes | yes |
| `myFree(ptr)` | — | frees |

### Transfers (never dereference GPU pointers from CPU)
- `myMemcpyH2D(dst, src, bytes)`
- `myMemcpyD2D(dst, src, bytes)`
- `myMemcpyD2H(dst, src, bytes)`
- `...Async` variants exist for the same three directions.
- `clearCuMem(ptr, bytes)` — device memset.
- `memMngr.useOnsite(bytes)` — borrow a device buffer in-place (no copy).

Example (`src/post_gpu/cdi.cc`, `CDI::allocateMem`):
```cpp
myCuMalloc(complexFormat, objectWave, row*column);
myCuMalloc(Real, patternData, row*column);
```
Example (`util/fftDemo.cpp`):
```cpp
Real* img = readImage(argv[1], row, col);
myCuDMalloc(Real, d_intensity, row*col);
myMemcpyH2D(d_intensity, img, row*col*sizeof(Real));
myFree(img);               // recycle host buffer
myCuDMalloc(complexFormat, d_amp, row*col*sq(oversampling));
```

## 3. CUDA & FFT initialization

Call order for a new image size (see `util/propagatorDemo.cpp`,
`src/post_gpu/cdi.cc`):
```cpp
init_cuda_image();                 // required once: allocates the render LUT
resize_cuda_image(rows, cols);     // set the working image size (re-call after any size change)
init_fft(rows, cols);              // build the cuFFT plan for this size
plt.init(rows, cols);              // set the plotter image size
```
- `resize_cuda_image(row, col, hei = 1)` — MUST be called before any `plt.*`
  call and after every change of grid dimensions.
- `init_cuda_image(rcolor = 65535, scale = 1)` — global color scale setup.
- `init_fft(rows, cols, batch = 1)` — cuFFT plan. `createPlan(&handle, row,
  col)` / `createPlan1d(&handle, n)` / `destroyPlan(handle)` for manual plans.
- `gpuerr()` — synchronize + check for device errors. Use this, never a direct
  `cudaDeviceSynchronize`.
- `getGPUFreeMem()` — query available device memory.

## 4. FFT operations (`cudaConfig.hpp`)

- `myFFT(complexFormat* in, complexFormat* out)` — 2-D forward FFT (in-place if
  `in == out`).
- `myIFFT(complexFormat* in, complexFormat* out)` — 2-D inverse FFT.
- `myFFTM(void* handle, in, out)` / `myIFFTM` — use an explicit plan handle.
- `myFFTR2C(Real* in, complexFormat* out)` — real→complex FFT.
- `myFFTC2R(complexFormat* in, Real* out)` — complex→real FFT.
- `applyNorm(data, factor)` — multiply whole array by a scalar (Real or
  complex). Used to implement the 1/N and 1/√N FFT conventions.
- `cudaConvertFO(T* data, T* out = 0, Real norm = 1)` — fftshift (move the
  DC / zero-frequency component to the center) and apply `norm`.
- `convertFOPhase(complexFormat* data, Real norm = 1)` — fftshift of a phase
  field with norm.

Example — a normalized forward FT of an oversampled field
(`util/fftDemo.cpp`):
```cpp
createWaveFront(d_intensity, 0, d_amp, row, col);   // build field from intensity
init_fft(row*oversampling, col*oversampling);
myFFT(d_amp, d_amp);
```
Example — the standard "centered spectrum" idiom (`src/post_gpu/cdi.cc`):
```cpp
myFFT(objectWave, patternWave);
convertFOPhase(patternWave, 1./sqrt(row*column));
```

## 5. Wavefront creation (`cudaConfig.hpp`, `misc.hpp`)

- `createWaveFront(Real* intensity, Real* phase, complexFormat* out, int row,
  int col, int shiftx = 0, int shifty = 0, Real phaseFactor = 0)` — build a
  complex field `out = sqrt(intensity) * exp(i*phase)`. `phase` may be `0` for
  zero phase.
- `createWaveFront(Real* intensity, Real* phase, complexFormat* out, Real
  oversampling, ...)` — same, with an oversampling factor instead of row/col.
- `createPlaneWave(complexFormat* out, Real peakX, Real peakY, Real amp)` —
  fill with a plane wave whose spectrum is peaked at `(peakX, peakY)`.
- `readComplexWaveFront(intensityFile, phaseFile, Real*& d_intensity,
  Real*& d_phase, int& row, int& col)` — read an intensity/phase pair to device
  memory.

Example (`src/post_gpu/cdi.cc`, `CDI::readObjectWave`):
```cpp
readComplexWaveFront(common.Intensity, common.Phase, d_intensity, d_phase, objrow, objcol);
row = objrow*oversampling; column = objcol*oversampling;
createWaveFront(d_intensity, d_phase, objectWave, objrow, objcol);
```

## 6. Propagation (`include/propagator.hpp`, `src/post_gpu/propagator.cc`)

The `propagator` object carries the physical grid/geometry; set the fields
before use. All distances and pixel sizes share one unit (microns in practice).

### Fields
```cpp
propagator prop;
prop.pixelsize = pixelsize;   // physical size of one grid pixel
prop.row     = row;           // grid rows
prop.column  = column;        // grid columns
prop.lambda  = lambda;        // wavelength
prop.distance = d;            // propagation distance
```

### Methods
- `fresnelPropagate(complexFormat* field)` — Fresnel (paraxial) propagation by
  `distance`, in-place. Sequence: `multiplyFresnelPhase` (input quadratic
  phase) → fftshift → FFT → `applyNorm(1/√(N))` → fftshift →
  `multiplyPatternPhase` (output quadratic phase).
- `angularSpectrumPropagate(in, out)` — exact paraxial angular-spectrum
  propagation by `distance` (out-of-place). FFT → `applyNorm(1/N)` → fftshift →
  `multiplyPropagatePhase(2πd/λ, (λ/(px·N))²)` → fftshift → IFFT. Evanescent
  components (spatial freq beyond the light cone) are zeroed.
- `angularSpectrumPropagateReverse(in, out)` — same as above with
  `distance → -distance` (back-propagation).
- `multiplyFresnelPhase(amp)` — multiply by the quadratic Fresnel phase
  `exp(i·π·λ·d·r²/(px²·N²))`.
- `removeFresnelPhase(amp)` — the inverse of `multiplyFresnelPhase`.
- `multiplyPatternPhase(amp)` / `multiplyPatternPhase_reverse(amp)` — multiply
  by the Fresnel output phase `exp(i·(π·px²·r²/(d·λ) + 2πd/λ − π/2))` and its
  inverse.
- `obliquePropagateFwd(in, out, thetaRad)` / `obliquePropagateBwd(...)` —
  propagation with the k-vector rotated into a tilted surface frame.

### Low-level phase kernels (direct access, used by the methods above)
- `multiplyPropagatePhase(complexFormat* amp, Real a, Real b)` — frequency-domain
  propagation phase `exp(i·a·sqrt(1 − b·r̂²))` where `a = z/λ` (or `2πd/λ` for
  the angular form) and `b = (λ/s)²`, `s` = image size in physical units.
- `multiplyFresnelPhase_Device(amp, phaseFactor)` — spatial-domain
  `exp(i·phaseFactor·r²)`.
- `multiplyPatternPhase_Device(amp, r_d_lambda, d_r_lambda)` — frequency-domain
  `exp(i·(r_d_lambda·r² + d_r_lambda))`.

Example — set up and run a straight + oblique propagation
(`util/obliqueDemo.cpp`):
```cpp
propagator prop;
prop.lambda = lambda; prop.distance = D; prop.pixelsize = pixelsize;
prop.row = N; prop.column = N;
prop.angularSpectrumPropagate(d_pupil, d_straight);
plt.plotComplexColor(d_straight, 0, 1, "straight_color");
prop.obliquePropagateFwd(d_pupil, d_oblique, theta);
plt.plotComplexColor(d_oblique, 0, 1, "oblique_color");
```
Example — the CDI setup (`src/post_gpu/cdi.cc`, `CDI::allocateMem`):
```cpp
propagate.row = row; propagate.column = column;
propagate.pixelsize = pixelsize; propagate.lambda = lambda;
propagate.distance = d;
...
if(isFresnel) propagate.multiplyFresnelPhase(objectWave);   // before the FFT
...
if(isFresnel) propagate.removeFresnelPhase(cuda_gkp1);      // after reconstruction
```

## 7. Amplitude / phase operations (`cudaConfig.hpp`)

- `getMod2(Real* mod2, complexFormat* amp, Real norm = 1)` — intensity `|amp|²·norm`.
- `getMod2(complexFormat* mod2, complexFormat* amp, Real norm = 1)` — complex intensity.
- `getMod(Real* mod, complexFormat* amp)` — amplitude `|amp|`.
- `getArg(Real* angle, complexFormat* amp)` — phase `atan2(imag, real)`.
- `getImag(Real* out, complexFormat* amp)`, `assignReal(Real* out, complexFormat* amp)`.
- `multiply<T1,T2>(T1* store, T1* source, T2* target)` — elementwise
  `store = source * target`.
- `multiplyReal(Real* store, complexFormat* a, complexFormat* b)` — real elementwise product.
- `multiplyConj(complexFormat* store, complexFormat* src, complexFormat* target)` — `src * conj(target)`.
- `multiplyRegular(complexFormat* store, complexFormat* src, complexFormat* target, Real alpha)`.
- `add(...)` — elementwise sum (Real and complex overloads).
- `addPhase(complexFormat* a, complexFormat* b, Real phi0 = 0)` — `a += b·e^{iφ0}`.
- `applyGaussMult(complexFormat* in, complexFormat* out, Real sigma, bool isFreq)` — Gaussian regularization.
- `regularize(complexFormat* object, Real sigma)`.
- `multiplyShift(complexFormat* wave, Real shiftx, Real shifty)` — spatial shift via phase.
- `forcePositive(data)` / `logorithm(in, out, k, b)` / `power(in, out, p, k, b)`.
- `invert(Real* data)` — `data = 1/data`.
- `bitMap`, `applyThreshold`, `ceiling`, `extendToComplex`, `forcePositive`.

Example (`src/post_gpu/cdi.cc`):
```cpp
getMod2(cuda_objMod, patternWave);
multiply(cuda_objMod, cuda_objMod, beamstop);   // elementwise product
```

## 8. Reductions (`include/cub_wrap.hpp`, `misc.hpp`)

- `findSum(Real*/complexFormat* in, int num = 0, void* out = NULL)` — array sum.
- `findRootSumSq(Real* in, ...)` — L2 norm.
- `findMax(Real* in, ...)` — max value; `findMaxIdx` — index of max.
- `findMod2Max(complexFormat* in, ...)` — max of `|amp|²`.
- `findSumReal(complexFormat* in, ...)` — sum of real parts.
- `findMiddle(complexFormat* in, int num = 0)` — value at the center pixel
  (`misc.hpp`); returns a `complexFormat` with the coordinate in `x`, `y`.
- `initCub()` — one-time init of the cub reductions.

Example (`src/post_gpu/cdi.cc`):
```cpp
Real maxs = findMax(patternData, row*column);
```

## 9. Image I/O (`include/imgio.hpp`)

- `Real* readImage(const char* name, int &row, int &col, void* cmalloc = 0)` —
  read an image into host memory, fill `row`/`col`. Supports png/tiff/bin.
- `Image3 readImage3(const char* name, ...)` — RGB image (`{r,g,b,rows,cols}`).
- `uint16_t* readRawPNG(const char* name, int* row, int* col)`.
- `Real* readPNG(const char* name, int &row, int &col)` (`misc.hpp`).
- `writeFloatImage(const char* name, void* data, int row, int col)` — write a
  float image (CPU buffer).
- `writeComplexImage(const char* name, void* data, int row, int col)`.
- `writePng(name, pix, width, height, bit_depth, colored)` / `writeJPEG(...)`.

Example (`util/fftDemo.cpp`):
```cpp
Real* img = readImage(argv[1], row, col);
myCuDMalloc(Real, d_intensity, row*col);
myMemcpyH2D(d_intensity, img, row*col*sizeof(Real));
myFree(img);
```

## 10. Plotting (`include/cuPlotter.hpp`, `src/post_gpu/cuPlotter.cc`)

`plt` is the global `cuPlotter` singleton. `plt.init(rows, cols, prefix =
"./")` sets the image size (and output prefix). Always call `resize_cuda_image`
with the same dimensions first.

### Mode enum
`MOD2` (|amp|²), `MOD` (|amp|), `REAL`, `IMAG`, `PHASE` (in [−π, π]), `PHASERAD`.

### Main plotting calls
- `plt.plotComplex(complexFormat* data, mode m = MOD, bool isFrequency = 0,
  Real decay = 1, const char* label = "default", bool islog = 0,
  bool isFlip = 0, bool isColor = 0, const char* caption = 0)`
  - `isFrequency` — center the DC component before display (fftshift view).
  - `decay` — intensity scale/normalization factor.
  - `label` — output file base name (`.png` appended unless color).
  - `islog` — log-scale the intensity.
  - `isFlip` — vertical flip.
  - `isColor` — render with a color map instead of grayscale.
- `plt.plotComplexColor(complexFormat* data, bool isFrequency = 0,
  Real decay = 1, const char* label = "default", bool islog = 0,
  bool isFlip = 0, bool phaseOnly = 0)`
  - Color wavefront: hue encodes phase, brightness encodes |amp| (scaled by
    `decay`, clamped to [0,1]). `phaseOnly` shows the phase as a full-saturation
    color wheel. Output is `.jpg`. This is the go-to for visualizing a field.
- `plt.plotFloat(Real* data, mode m, isFrequency, decay, label, islog, isFlip,
  isColor, caption)` — like `plotComplex` for a real buffer.
- `plt.saveComplex(complexFormat* data, label)` / `plt.saveFloat(Real* data,
  label)` — save raw data files.

### Video
- `int handle = plt.initVideo(filename, fps = 24, preview = 0, online = 0)`.
- Set `plt.toVideo = handle;`, plot frames, then `plt.saveVideo(handle)`.

Example — color wavefronts (`util/obliqueDemo.cpp`):
```cpp
plt.plotComplexColor(d_straight, 0, 1, "straight_color");
plt.plotComplex(d_oblique, MOD, 0, 1, "oblique_intensity.png", 1, 0, 1);
```
Example — frequency-domain intensity (`src/post_gpu/cdi.cc`):
```cpp
plt.plotComplex(patternWave, PHASE, 1, 1, "init_pattern_phase", 0);
```
Example — in-loop video frames (`src/post_gpu/cdi.cc`):
```cpp
plt.toVideo = vidhandle;
plt.plotComplexColor(cuda_gkp1, 0, sqrt(row*column), ("recon_intensity"+to_string(iter)).c_str(), 0, isFlip);
plt.toVideo = -1;
```

## 11. Config parsing (`include/readConfig.hpp`)

Configs are read from `.cfg` files using an X-macro pattern. Variables are
declared in four lists; each is expanded by the corresponding `DEC*` macro:

- `INTVAR(F)` → `int x = y;`
- `BOOLVAR(F)` → `bool x = y;`
- `REALVAR(F)` → `double x = y;`
- `STRVAR(F)` → `const char* x = y;`

A class that inherits `readConfig` and calls `readConfig(configfile)` gets all
these members populated from the file. The `AlgoParser` class parses algorithm
formulas like `"1000*(20*HIO+10*RAAR)"` into a sequence of
`(count, algorithm)` pairs; `next()` returns the next step and `restart()`
rewinds.

Example (`src/post_gpu/cdi.cc`):
```cpp
class CDI : public readConfig { ... };
CDI::CDI(const char* configfile):readConfig(configfile){ ... }
```
See `html/default/cdi.cfg` for a full annotated config file.

## 12. Other utilities (`cudaConfig.hpp`, `misc.hpp`)

### Geometric / array ops
- `crop<T>(T* src, T* dest, int row, int col, Real midx = 0, Real midy = 0)` —
  extract a centered `row×col` crop.
- `pad<T>(T* src, T* dest, int row, int col, int shiftx = 0, int shifty = 0)` —
  embed into a larger zero-padded array.
- `resize<T>(const T* in, T* out, int in_width, int in_height)` — bilinear resample.
- `paste(...)` — paste a sub-image at a position.
- `cropinner` / `padinner` — normalized inner crop/pad.
- `refine<T>(T* src, T* dest, int refinement)`.
- `rotate90` / `rotate` / `transpose` / `flipx` / `flipy`.

### Masks & random
- `createMask(Real* data, T* spt, bool isFrequency = 0)` / `addMask` /
  `createMaskBar` / `applyMask` / `applyMaskBar`.
- `createCircleMask(Real* data, Real x0, Real y0, Real r, bool isFreq = 0)`.
- `createBeamStop(int row, int column, int beamStopSize)` — returns a host beamstop.
- `createMaskMap(...)`.
- `initRand(void* state, unsigned long long seed)`.
- `applyRandomPhase(complexFormat* wave, Real* beamstop, void* state)` — multiply
  by a random phase (random phase object), optionally zeroing the beamstop.
- `randMask(char* mask, void* state, Real ratio = .5)`.

### Drawing
- `drawCircle(complexFormat*/Real* img, Real cx, Real cy, Real r, Real w, value)`.
- `createColorbar(complexFormat* output)`.

### Spectral / other
- `getRealSpectrum(...)` / `getNormSpectrum(...)` — load a spectral response.
- `peakGaussSum` / `peakGaussRatio` — Gaussian peak analysis.
- `phaseUnwrapping(complexFormat* d_wrapped_phase)`.
- `solve_poisson_frequency_domain(complexFormat* d_fft_data)`.
- `cart2polar_kernel(...)` / `edgeReduce(...)` — polar transform + line reduction.
- `applyGaussConv(...)`.
- `findPeaks` / `getPeakEnergies` / `refinePeaks` / `createPeakGauss` (`misc.hpp`).

Example — crop a centered region for display (`util/propagatorDemo.cpp`):
```cpp
crop(d_propagatedwave, d_crop, rowi, coli, mid.x, mid.y);
plt.plotComplexColor(d_crop, 0, 0.3, ("test_" + to_string(...)).c_str(), 0, 0);
```

## 13. Typical workflow

Putting it together, a minimal "load → propagate → plot" program follows this
skeleton (assembled from `util/fftDemo.cpp`, `util/propagatorDemo.cpp` and
`util/obliqueDemo.cpp`):

```cpp
#include "fmt/core.h"
#include "imgio.hpp"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "propagator.hpp"
#include "misc.hpp"
#include "cub_wrap.hpp"
using namespace std;

int main(){
  init_cuda_image();
  int N = 1024;
  Real pixelsize = 2.0, lambda = 0.6328, D = 3e3;

  resize_cuda_image(N, N);
  init_fft(N, N);
  plt.init(N, N);

  // Build an input field on the host, upload it.
  myDMalloc(complexFormat, h, N*N);
  /* fill h ... */
  myCuDMalloc(complexFormat, d_in, N*N);
  myMemcpyH2D(d_in, h, N*N*sizeof(complexFormat));
  myFree(h);

  // Propagate.
  propagator prop;
  prop.pixelsize = pixelsize; prop.row = N; prop.column = N;
  prop.lambda = lambda; prop.distance = D;
  myCuDMalloc(complexFormat, d_out, N*N);
  prop.angularSpectrumPropagate(d_in, d_out);

  // Plot the result as a color wavefront.
  plt.plotComplexColor(d_out, 0, 1, "propagated_color");
  plt.plotComplex(d_out, MOD, 0, 1, "propagated_intensity.png", 1, 0, 1);
}
```

### Build & run
Each file under `util/<name>.cpp` builds to `bin/<name>_run` (auto-globbed by the
root `CMakeLists.txt`). Build and run from a scratch workspace because configs
use a relative `outputDir`:

```bash
source env.sh                                  # sets CDI_DIR, PATH, LD_LIBRARY_PATH
ninja -C build <name>_run                     # build one target
cd /path/to/workspace && source $CDI_DIR/env.sh
$CDI_DIR/bin/<name>_run [args]
```
