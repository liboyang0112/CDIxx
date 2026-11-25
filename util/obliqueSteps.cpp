#include "fmt/core.h"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "propagator.hpp"
#include "misc.hpp"
#include <math.h>
#include <string>
using namespace std;

// Host wrapper exported by libcudaLib (defined in src/gpu/cudaConfig.cu)
void obliqueRotatePhase(complexFormat* input, complexFormat* output, Real a, Real b, Real sinT, Real cosT, int sign);

// FWHM along x (columns) through row, or along y (rows) through column
static Real fwhmX(Real* h, int N, int row){
  Real peak = 0;
  for(int j = 0; j < N; j++) peak = fmaxf(peak, h[row*N+j]);
  Real half = peak/2;
  int j0 = N/2; while(j0 > 0 && h[row*N+j0] > half) j0--;
  int j1 = N/2; while(j1 < N-1 && h[row*N+j1] > half) j1++;
  return j1 - j0;
}
static Real fwhmY(Real* h, int N, int col){
  Real peak = 0;
  for(int i = 0; i < N; i++) peak = fmaxf(peak, h[i*N+col]);
  Real half = peak/2;
  int i0 = N/2; while(i0 > 0 && h[i0*N+col] > half) i0--;
  int i1 = N/2; while(i1 < N-1 && h[i1*N+col] > half) i1++;
  return i1 - i0;
}
static void reportWidth(complexFormat* d_field, int N, const char* name){
  myDMalloc(complexFormat, h_field, N*N);
  myMemcpyD2H(h_field, d_field, N*N*sizeof(complexFormat));
  myDMalloc(Real, h_int, N*N);
  for(int i = 0; i < N*N; i++)
    h_int[i] = h_field[i].x*h_field[i].x + h_field[i].y*h_field[i].y;
  fmt::println("{}: FWHM_x={:.1f}px  FWHM_y={:.1f}px  ratio={:.4f}",
               name, fwhmX(h_int, N, N/2), fwhmY(h_int, N, N/2), fwhmX(h_int, N, N/2)/fwhmY(h_int, N, N/2));
  myFree(h_int);
  myFree(h_field);
}

int main(int argc, char* argv[]){
  int N = 1024;
  Real pixelsize = 2.0;
  Real lambda = 0.6328;
  Real D = 3e3;
  Real theta = 0.1;
  float w0 = 128.0;
  if(argc > 1) theta = stof(argv[1]);
  if(argc > 2) pixelsize = stof(argv[2]);
  if(argc > 3) w0 = stof(argv[3]);
  if(argc > 4) D = stof(argv[4]);

  init_cuda_image();
  resize_cuda_image(N, N);
  plt.init(N, N);
  init_fft(N, N);

  fmt::println("params: theta={:.4f} rad, pixelsize={:.3f} um, w0={:.0f}px, D={:.0f}um",
               theta, pixelsize, w0, D);
  fmt::println("k-space FOV: +/- lambda/(2px) = {:.3f} (k0 units) = {:.1f} deg; expected stretch 1/cos(theta) = {:.4f}",
               lambda/(2*pixelsize), atanf(lambda/(2*pixelsize))*180.0f/M_PI, 1.0f/cosf(theta));

  // ---------- Step 1: initial beam (Gaussian) at the pupil plane ----------
  myCuDMallocClean(complexFormat, d_pupil, N*N);
  myDMalloc(complexFormat, h_pupil, N*N);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      float x = (j - N/2);
      float y = (i - N/2);
      float r2 = x*x + y*y;
      float amp = expf(-r2/(w0*w0));
      h_pupil[i*N+j] = complexFormat(amp, 0);
    }
  }
  myMemcpyH2D(d_pupil, h_pupil, N*N*sizeof(complexFormat));
  myFree(h_pupil);

  plt.plotComplex(d_pupil, MOD, 0, 1, "step1_initial_beam.png", 1, 0, 1);
  plt.plotComplexColor(d_pupil, 0, 1, "step1_initial_beam");

  // ---------- Step 2: k-vectors (angular spectrum of the pupil) ----------
  myCuDMalloc(complexFormat, d_kv, N*N);
  myFFT(d_pupil, d_kv);
  applyNorm(d_kv, 1./(N*N));
  cudaConvertFO(d_kv);  // center the spectrum

  plt.plotComplex(d_kv, MOD, 0, 1, "step2_kvectors.png", 1, 0, 1);
  plt.plotComplex(d_kv, PHASE, 0, 1, "step2_kvectors_phase.png", 0, 0, 1);

  // dump centered angular spectrum
  {
    myDMalloc(complexFormat, h_dump, N*N);
    myMemcpyD2H(h_dump, d_kv, N*N*sizeof(complexFormat));
    FILE* f = fopen("step2_kv_centered.cplx", "wb");
    fwrite(h_dump, sizeof(complexFormat), N*N, f);
    fclose(f);
    myFree(h_dump);
  }
  // ---------- Step 3: transformed k-vectors (rotated into oblique surface frame) ----------
  myCuDMalloc(complexFormat, d_kvt, N*N);
  Real lambda_over_imagesize = lambda/(pixelsize*N);
  Real a = 2*M_PI*D/lambda;
  Real b = lambda_over_imagesize*lambda_over_imagesize;
  obliqueRotatePhase(d_kv, d_kvt, a, b, sinf(theta), cosf(theta), +1);

  // cancel the constant k-shift of the gather: the beam lands at kx = sin(theta)
  // (k0 units) = s_px off the grid center; remove the integer part here (exact),
  // the residual <1 px is removed as a phase after the IFFT
  const Real s_px = sinf(theta)*pixelsize*N/lambda;
  const int s_int = (int)floorf(s_px);
  const Real s_frac = s_px - (Real)s_int;
  if(s_int != 0){
    myDMalloc(complexFormat, h_kvt, N*N);
    myDMalloc(complexFormat, h_kvt_s, N*N);
    myMemcpyD2H(h_kvt, d_kvt, N*N*sizeof(complexFormat));
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++){
        int src = (j + s_int) % N;
        if(src < 0) src += N;
        h_kvt_s[i*N+j] = h_kvt[i*N+src];
      }
    myMemcpyH2D(d_kvt, h_kvt_s, N*N*sizeof(complexFormat));
    myFree(h_kvt_s);
    myFree(h_kvt);
  }

  plt.plotComplex(d_kvt, MOD, 0, 1, "step3_transformed_kvectors.png", 1, 0, 1);
  plt.plotComplex(d_kvt, PHASE, 0, 1, "step3_transformed_kvectors_phase.png", 0, 0, 1);
  // dump rotated (centered) spectrum
  {
    myDMalloc(complexFormat, h_dump, N*N);
    myMemcpyD2H(h_dump, d_kvt, N*N*sizeof(complexFormat));
    FILE* f = fopen("step3_kvt_centered.cplx", "wb");
    fwrite(h_dump, sizeof(complexFormat), N*N, f);
    fclose(f);
    myFree(h_dump);
  }

  // ---------- Step 4: wavefront in oblique surface (IFFT of transformed k-vectors) ----------
  cudaConvertFO(d_kvt);  // uncenter before IFFT
  myCuDMalloc(complexFormat, d_surf, N*N);
  myIFFT(d_kvt, d_surf);

  // remove the residual (<1 px) ramping phase from the fractional part of the k-shift
  // (kx is the column axis on this grid, per the obliqueRotatePhase kernel)
  if(fabsf(s_frac) > 1e-6f){
    myDMalloc(complexFormat, h_surf, N*N);
    myDMalloc(Real, h_c, N);
    myDMalloc(Real, h_s, N);
    for(int j = 0; j < N; j++){
      float ph = -2*M_PI*s_frac*(j - N/2)/N;
      h_c[j] = cosf(ph);
      h_s[j] = sinf(ph);
    }
    myMemcpyD2H(h_surf, d_surf, N*N*sizeof(complexFormat));
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++){
        complexFormat v = h_surf[i*N+j];
        h_surf[i*N+j] = complexFormat(v.x*h_c[j] - v.y*h_s[j], v.x*h_s[j] + v.y*h_c[j]);
      }
    myMemcpyH2D(d_surf, h_surf, N*N*sizeof(complexFormat));
    myFree(h_surf);
    myFree(h_c);
    myFree(h_s);
  }

  plt.plotComplex(d_surf, MOD, 0, 1, "step4_oblique_surface.png", 1, 0, 1);
  plt.plotComplexColor(d_surf, 0, 1, "step4_oblique_surface");

  // ---------- Step 5 (reference): straight propagation to a flat surface ----------
  myCuDMalloc(complexFormat, d_straight, N*N);
  propagator prop;
  prop.lambda = lambda;
  prop.distance = D;
  prop.pixelsize = pixelsize;
  prop.row = N;
  prop.column = N;
  prop.angularSpectrumPropagate(d_pupil, d_straight);

  plt.plotComplex(d_straight, MOD, 0, 1, "step5_straight_surface.png", 1, 0, 1);
  plt.plotComplexColor(d_straight, 0, 1, "step5_straight_surface");

  // dump oblique surface field (float32 complex) for offline analysis
  {
    myDMalloc(complexFormat, h_dump, N*N);
    myMemcpyD2H(h_dump, d_surf, N*N*sizeof(complexFormat));
    FILE* f = fopen("step4_surf_field.cplx", "wb");
    fwrite(h_dump, sizeof(complexFormat), N*N, f);
    fclose(f);
    myFree(h_dump);
  }
  fmt::println("Done. Beam widths on the surfaces:");
  reportWidth(d_straight, N, "straight");
  reportWidth(d_surf,     N, "oblique");
  fmt::println("Done. Step images written to CWD.");
  return 0;
}
