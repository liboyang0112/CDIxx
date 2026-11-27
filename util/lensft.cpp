// lensft.cpp
// Lens Fourier-transform propagation demo, built on the CDIxx CUDA operation API.
//
// Physical layout (transverse axis = x/y, propagation axis = z):
//
//      object          lens (f)          detector
//   (z=0)  --d1-->  (z=d1)   --d2-->  (z=d1+d2)
//
//   * The field is sampled on a single fixed N x N grid (pixel size = dx,
//     "granularity"). The object occupies the central M x M sub-region
//     (M = N/2), i.e. an oversampling ratio of 2 (zero-padding for the FT).
//   * object -> lens : Fresnel diffraction over distance d1.
//   * lens         : multiply by the spherical-surface lens transfer function
//                     H(r) = exp(-i * k * (sqrt(f^2 + r^2) - f)),  k = 2*pi/lambda
//                     (exact converging spherical wave; the quadratic thin lens
//                      is its r << f paraxial limit).
//   * lens -> detector : angular-spectrum propagation over distance d2
//                     (grid size and granularity unchanged).
//
//   A perfect Fourier transform of the object is obtained when d1 = d2 = f
//   (object in the front focal plane, detector in the back focal plane).
//   Here d1 and d2 are "floated" (independent) to demonstrate the general,
//   possibly defocused, case.
//
//   The wavefronts at every stage are rendered with plt.plotComplexColor
//   (hue = phase, brightness = amplitude).
//
// Build:  ninja -C build lensft_run
// Run:    cd /path/to/workspace && $CDI_DIR/bin/lensft_run

#include "fmt/core.h"
#include "cudaConfig.hpp"   // memory pool, FFT, wavefront, phase, plot helpers
#include "cuPlotter.hpp"    // plt
#include "propagator.hpp"   // propagator (fresnel / angular-spectrum)
#include "misc.hpp"
#include "cub_wrap.hpp"     // findMod2Max
#include <math.h>
using namespace std;

// ---------------------------------------------------------------------------
// Geometry / optical parameters (all lengths in the same unit, microns here).
// ---------------------------------------------------------------------------
static const int   N        = 2048;     // grid size (both dims), same as ssBeam (power of 2; not a multiple of the period 6, so the Bragg peaks are slightly leaked)
static const int   oversampling = 2;    // oversampling ratio
static const int   M        = N / oversampling;  // object size in pixels
static const Real  dx       = 2.0;      // grid pixel size (granularity), microns
static const Real  lambda   = 0.6328;      // wavelength, microns

// Focal length for a correctly scaled FT on a detector grid that keeps the
// same granularity:  f = N * dx^2  (from dx_det = f/(N*dx) == dx).
static const Real  f        = 40e3;   // 2048 um

// Distances:  d1 = object -> lens,  d2 = lens -> detector.
// Perfect FT: d1 = d2 = f.  We also run a "floated" (defocused) case.
static const Real  d1_perfect = f;
static const Real  d2_perfect = f;
static const Real  d1_floated = 1.5 * f;   // object not in the front focal plane
static const Real  d2_floated = 0.7 * f;   // detector not in the back focal plane

// Object wavefront: the pupilFuncSS/ssBeam dot lattice (period 3, unit cell
// (0,0,0;0,1,0;0,0,0)) OVERSAMPLED BY 2 -- each dot becomes a 2x2 block, so the
// unit cell is 6x6 with a 2x2 block of 1s at its center:
//     0 0 0 0 0 0
//     0 0 0 0 0 0
//     0 0 1 1 0 0
//     0 0 1 1 0 0
//     0 0 0 0 0 0
//     0 0 0 0 0 0
// gated by a circular aperture of diameter pupilsize.  Pure real (0/1), zero phase.
// `pupilsize` = M = N/2 (the object region, oversampling 2).
static const Real  pupilsize = M;

// Build the oversampled dot lattice (N x N) on the host: 1 where (i mod 6) and
// (j mod 6) are in {2,3} (the 2x2 block at the center of each 6x6 cell) and the
// pixel is inside the circular aperture; 0 otherwise.
static Real* buildObjectField(){
  myDMalloc(Real, h_int, N*N);
  Real ap2 = (pupilsize*pupilsize)/4.0;   // aperture radius^2
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      int shiftx = j - N/2;
      int shifty = i - N/2;
      Real val = 0.0;
      if(shiftx*shiftx + shifty*shifty <= ap2){
        int xm = j % 6;
        int ym = i % 6;
        if((xm==2 || xm==3) && (ym==2 || ym==3)) val = 1.0;
      }
      h_int[i*N + j] = val;
    }
  }
  return h_int;
}

// Build the spherical-surface lens transfer function (N x N) on the host.
//
//   H(x,y) = exp(-i * k * (sqrt(f^2 + r^2) - f)),   k = 2*pi/lambda, r^2 = x^2 + y^2
//
// This is the exact phase of a converging spherical wave focused to the point at
// distance f, i.e. a physical spherical lens surface.  The paraxial (quadratic)
// thin lens is its r << f limit:
//   sqrt(f^2 + r^2) - f  =  r^2/(2f) - r^4/(8f^3) + r^6/(16f^5) - ...
// so the spherical lens adds the higher-order (spherical-aberration) terms
// r^4, r^6, ... that the quadratic phase exp(-i*pi*r^2/(f*lambda)) drops.
static complexFormat* buildLensTF(){
  myDMalloc(complexFormat, h_lens, N*N);
  Real k = 2.0 * M_PI / lambda;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      Real px = (j - N/2.0) * dx;
      Real py = (i - N/2.0) * dx;
      Real r2 = px*px + py*py;
      Real path = sqrtf(f*f + r2) - f;   // exact spherical-wave path delay
      Real arg  = k * path;
      h_lens[i*N + j] = complexFormat(cosf(arg), -sinf(arg));
    }
  }
  return h_lens;
}

// Plot a complex wavefront with a well-exposed color scale (decay = 1/max|amp|).
static void plotWavefront(complexFormat* field, const char* label){
  Real mod2max = findMod2Max(field, N*N);
  Real decay = (mod2max > 0) ? 1.0 / sqrtf(mod2max) : 1.0;
  plt.plotComplexColor(field, 0, decay, label, 0, 0);
}

// d_lensTF is the global thin-lens transfer function buffer (device).
static complexFormat* d_lensTF = 0;

// Run the full object -> lens -> detector pipeline for given distances.
//   in  : input object field (N x N), untouched (copied before propagation).
//   out : field at the detector (N x N).
//   atLens / afterLens : optional buffers to capture intermediate wavefronts.
static void runPipeline(complexFormat* in, Real d1, Real d2,
                        complexFormat* out,
                        complexFormat* atLens = 0, complexFormat* afterLens = 0){
  propagator prop;
  prop.pixelsize = dx;
  prop.row = N;
  prop.column = N;
  prop.lambda = lambda;

  // 1. object -> lens : Fresnel diffraction over d1 (in place).
  myCuDMalloc(complexFormat, work, N*N);
  myMemcpyD2D(work, in, N*N*sizeof(complexFormat));
  prop.distance = d1;
  prop.fresnelPropagate(work);
  if(atLens) myMemcpyD2D(atLens, work, N*N*sizeof(complexFormat));

  // 2. lens : multiply by the thin-lens transfer function (in place).
  multiply(work, work, d_lensTF);
  if(afterLens) myMemcpyD2D(afterLens, work, N*N*sizeof(complexFormat));

  // 3. lens -> detector : angular-spectrum propagation over d2.
  prop.distance = d2;
  prop.angularSpectrumPropagate(work, out);

  myCuFree(work);
}

int main(){
  fmt::println("=== Lens FT propagation demo ===");
  fmt::println("grid N={} (object M={}), dx={} um, lambda={} um", N, M, dx, lambda);
  fmt::println("f={} um | perfect d1=d2=f | floated d1={} um, d2={} um",
               f, d1_floated, d2_floated);

  // --- CUDA / FFT / plotter initialization ---------------------------------
  init_cuda_image();
  resize_cuda_image(N, N);
  init_fft(N, N);
  plt.init(N, N);

  // --- object field: dot lattice oversampled by 2 (2x2 blocks) -------------
  Real* h_int = buildObjectField();
  myCuDMalloc(Real, d_objint, N*N);
  myMemcpyH2D(d_objint, h_int, N*N*sizeof(Real));
  myFree(h_int);
  myCuDMalloc(complexFormat, d_obj, N*N);
  extendToComplex(d_objint, d_obj);   // real 0/1 field -> complex (zero phase)
  myCuFree(d_objint);

  // --- thin-lens transfer function -----------------------------------------
  complexFormat* h_lens = buildLensTF();
  d_lensTF = (complexFormat*)memMngr.borrowCache(N*N*sizeof(complexFormat));
  myMemcpyH2D(d_lensTF, h_lens, N*N*sizeof(complexFormat));
  myFree(h_lens);

  // --- buffer for intermediate / output wavefronts -------------------------
  myCuDMalloc(complexFormat, d_atLens,    N*N);
  myCuDMalloc(complexFormat, d_afterLens, N*N);
  myCuDMalloc(complexFormat, d_detPerfect, N*N);
  myCuDMalloc(complexFormat, d_detFloated, N*N);

  // --- stage 0: object -----------------------------------------------------
  plotWavefront(d_obj, "0_object");

  // --- perfect FT case (d1 = d2 = f) ---------------------------------------
  runPipeline(d_obj, d1_perfect, d2_perfect, d_detPerfect, d_atLens, d_afterLens);
  fmt::println("--- perfect case (d1=d2=f) ---");
  plotWavefront(d_atLens,    "1_at_lens_perfect");
  plotWavefront(d_afterLens, "2_after_lens_perfect");
  plotWavefront(d_detPerfect,"3_detector_perfect");
  // Intensity-only view of the FT (amplitude, no phase color).
  Real detmax = findMod2Max(d_detPerfect, N*N);
  Real idcay = (detmax > 0) ? 1.0/sqrtf(detmax) : 1.0;
  plt.plotComplex(d_detPerfect, MOD, 0, idcay, "3_detector_perfect_intensity.png", 1, 0, 1);
  // Full phase color-wheel view (phaseOnly ignores amplitude) so the
  // wavefront's phase structure is visible across the whole grid.
  plt.plotComplexColor(d_detPerfect, 0, 1, "3_detector_perfect_phaseonly", 0, 0, 1);

  // --- direct-FFT reference for the perfect case (d1 = d2 = f) ------------
  // In this configuration the field at the detector is the Fourier transform
  // of the object, up to a residual quadratic (Ewald) phase.  That phase is
  // purely quadratic, so it cancels in the intensity:
  //     |detector|^2  ==  |FFT(object)|^2    (up to a constant scale).
  // With f = N*dx^2 and lambda = 1 the detector pixel index equals the FFT
  // frequency index, so the two align pixel-by-pixel.  Compare them with a
  // scale-invariant metric and show the residual difference.
  myCuDMalloc(complexFormat, d_directFFT, N*N);
  myFFT(d_obj, d_directFFT);
  cudaConvertFO(d_directFFT);           // center the DC, matching the detector view

  myCuDMalloc(Real, d_int_det, N*N);
  myCuDMalloc(Real, d_int_fft, N*N);
  myCuDMalloc(Real, d_prod,   N*N);
  getMod2(d_int_det, d_detPerfect);
  getMod2(d_int_fft, d_directFFT);

  // Normalized cross-correlation of the two intensities (scale-invariant).
  multiply(d_prod, d_int_det, d_int_fft);
  Real dot     = findSum(d_prod, N*N);
  Real normDet = findRootSumSq(d_int_det, N*N);
  Real normFFT = findRootSumSq(d_int_fft, N*N);
  Real ncc     = dot / (normDet * normFFT);
  fmt::println("NCC(|detector|^2, |FFT(object)|^2) = {:.4f}   (1.0 = identical)", ncc);

  // Reference plots of the exact discrete FT.
  Real fftmax = findMod2Max(d_directFFT, N*N);
  Real ftdcay = (fftmax > 0) ? 1.0/sqrtf(fftmax) : 1.0;
  plt.plotComplex(d_directFFT, MOD, 0, ftdcay, "5_direct_fft_intensity.png", 1, 0, 1);
  plt.plotComplexColor(d_directFFT, 0, ftdcay, "5_direct_fft_color", 0, 0);

  // Normalized |detector - FFT|: where the paraxial pipeline deviates from the
  // exact discrete FT (concentrated at high spatial frequency / the edges).
  myCuDMalloc(Real, d_diff, N*N);
  Real maxDet = findMax(d_int_det, N*N);
  Real maxFFT = findMax(d_int_fft, N*N);
  applyNorm(d_int_det, 1.0 / maxDet);
  applyNorm(d_int_fft, 1.0 / maxFFT);
  add(d_diff, d_int_det, d_int_fft, -1.0);   // d_diff = det - fft
  plt.plotFloat(d_diff, MOD, 0, 1, "6_diff_directfft_vs_perfect.png", 1, 0, 1);

  // --- floated (defocused) case (d1 != d2 != f) ----------------------------
  runPipeline(d_obj, d1_floated, d2_floated, d_detFloated);
  fmt::println("--- floated case (d1=1.5f, d2=0.7f) ---");
  plotWavefront(d_detFloated, "4_detector_floated");
  Real fmax = findMod2Max(d_detFloated, N*N);
  Real fdcay = (fmax > 0) ? 1.0/sqrtf(fmax) : 1.0;
  plt.plotComplex(d_detFloated, MOD, 0, fdcay, "4_detector_floated_intensity.png", 1, 0, 1);

  fmt::println("Done. Wavefronts (color) and FT intensities written to:");
  fmt::println("  0_object.jpg / 1_at_lens_perfect.jpg / 2_after_lens_perfect.jpg");
  fmt::println("  3_detector_perfect.jpg + _intensity.png + _phaseonly.jpg");
  fmt::println("  5_direct_fft_intensity.png + 5_direct_fft_color.jpg");
  fmt::println("  6_diff_directfft_vs_perfect.png  (normalized difference)");
  fmt::println("  4_detector_floated.jpg + _intensity.png");

  gpuerr();
  return 0;
}
