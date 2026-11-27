#include "fmt/core.h"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "propagator.hpp"
#include "misc.hpp"
#include <math.h>
#include <string>
using namespace std;

// Sine-lens k-transform kernel from src/gpu/patternCorrection.cu (libcudaLib)
extern void reflCorrectSineKernel(Real* input, Real* output, Real* beamstop, float sin_t, float cos_t, float a);

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
  fmt::println("{}: FWHM_x={:.2f}px  FWHM_y={:.2f}px  ratio={:.5f}  peak={:.5f}",
               name, fwhmX(h_int, N, N/2), fwhmY(h_int, N, N/2),
               fwhmX(h_int, N, N/2)/fwhmY(h_int, N, N/2), h_int[N/2*N+N/2]);
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

  const float st = sinf(theta), ct = cosf(theta);
  const float sb = (float)(lambda/(pixelsize*N));  // k-grid spacing (k0 units)
  const float ap = (float)(2.0*M_PI*D/lambda);     // propagation phase coeff

  fmt::println("params: theta={:.4f} rad ({:.2f} deg), pixelsize={:.3f} um, w0={:.0f}px, D={:.1f} um",
               theta, theta*180.0f/M_PI, pixelsize, w0, D);
  fmt::println("k-grid: sb={:.3e} (k0 units), half-FOV={:.4f}; 1/cosT={:.5f}", sb, N/2.0f*sb, 1.0f/ct);

  // ---------- Steps 1-2: Gaussian pupil -> centered angular spectrum ----------
  myCuDMallocClean(complexFormat, d_pupil, N*N);
  myDMalloc(complexFormat, h_pupil, N*N);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      float x = (j - N/2);
      float y = (i - N/2);
      h_pupil[i*N+j] = complexFormat(expf(-(x*x+y*y)/(w0*w0)), 0);
    }
  }
  myMemcpyH2D(d_pupil, h_pupil, N*N*sizeof(complexFormat));
  myFree(h_pupil);

  myCuDMalloc(complexFormat, d_kv, N*N);
  myFFT(d_pupil, d_kv);
  applyNorm(d_kv, 1./(N*N));
  cudaConvertFO(d_kv);  // center

  myDMalloc(complexFormat, h_kv, N*N);
  myMemcpyD2H(h_kv, d_kv, N*N*sizeof(complexFormat));

  // ---------- Path A: current de-tilt gather (verbatim from obliqueSteps.cpp) ----------
  myDMalloc(complexFormat, h_A, N*N);
  for(int r = 0; r < N; r++){
    const float ky = (r - N/2) * sb;
    for(int c = 0; c < N; c++){
      const float kxt = (c - N/2) * sb + st;
      const float disc = 1.0f - kxt*kxt - ky*ky;
      if(disc < 0){ h_A[r*N+c] = complexFormat(0,0); continue; }
      const float kzt = sqrtf(disc);
      const float kxs = kxt*ct - kzt*st;
      const float ysrc = kxs/sb + (float)(N/2);
      const int c0 = (int)floorf(ysrc) - 1;
      const float fc = ysrc - (float)(int)floorf(ysrc);
      if(c0 < 0 || c0+3 >= N){ h_A[r*N+c] = complexFormat(0,0); continue; }
      const float wc0 = 0.5f*(-fc + 2*fc*fc - fc*fc*fc);
      const float wc1 = 0.5f*(2 - 5*fc*fc + 3*fc*fc*fc);
      const float wc2 = 0.5f*(fc + 4*fc*fc - 3*fc*fc*fc);
      const float wc3 = 0.5f*(-fc*fc + fc*fc*fc);
      const float re = wc0*h_kv[r*N+c0].x + wc1*h_kv[r*N+c0+1].x + wc2*h_kv[r*N+c0+2].x + wc3*h_kv[r*N+c0+3].x;
      const float im = wc0*h_kv[r*N+c0].y + wc1*h_kv[r*N+c0+1].y + wc2*h_kv[r*N+c0+2].y + wc3*h_kv[r*N+c0+3].y;
      const float kzs = sqrtf(fmaxf(1.0f - kxs*kxs - ky*ky, 0.0f));
      const float knorm = kzs*ct - kxs*st;
      float sr, cr;
      sincosf(ap * kzs, &sr, &cr);
      const float wj = kzs / (knorm + 1e-12f);
      h_A[r*N+c] = complexFormat(wj*(re*cr - im*sr), wj*(re*sr + im*cr));
    }
  }

  // ---------- Path B: same k-field through reflCorrectSineKernel (no Ewald) ----------
  myCuDMalloc(Real, d_re, N*N); myCuDMalloc(Real, d_im, N*N);
  myCuDMalloc(Real, d_ore, N*N); myCuDMalloc(Real, d_oi, N*N);
  myCuDMallocClean(Real, d_bs, N*N);
  myDMalloc(Real, h_re, N*N); myDMalloc(Real, h_im, N*N);
  for(int i = 0; i < N*N; i++){ h_re[i] = h_kv[i].x; h_im[i] = h_kv[i].y; }
  myMemcpyH2D(d_re, h_re, N*N*sizeof(Real));
  myMemcpyH2D(d_im, h_im, N*N*sizeof(Real));
  reflCorrectSineKernel(d_re, d_ore, d_bs, st, ct, sb);
  reflCorrectSineKernel(d_im, d_oi, d_bs, st, ct, sb);
  myDMalloc(complexFormat, h_B, N*N);
  myMemcpyD2H(h_re, d_ore, N*N*sizeof(Real));
  myMemcpyD2H(h_im, d_oi, N*N*sizeof(Real));
  for(int i = 0; i < N*N; i++) h_B[i] = complexFormat(h_re[i], h_im[i]);

  // ---------- Path C: B * wj * e^{i*ap*kzs} (add back the Jacobian + propagation phase) ----------
  // If Newton gather == closed-form gather, C must equal A pixel-by-pixel.
  myDMalloc(complexFormat, h_C, N*N);
  float wjMin = 1e30f, wjMax = -1e30f, maxA = 0;
  for(int i = 0; i < N*N; i++) maxA = fmaxf(maxA, hypotf(h_A[i].x, h_A[i].y));
  for(int r = 0; r < N; r++){
    const float ky = (r - N/2) * sb;
    for(int c = 0; c < N; c++){
      const float kxt = (c - N/2) * sb + st;
      const float disc = 1.0f - kxt*kxt - ky*ky;
      if(disc < 0){ h_C[r*N+c] = complexFormat(0,0); continue; }
      const float kzt = sqrtf(disc);
      const float kxs = kxt*ct - kzt*st;
      const float kzs = sqrtf(fmaxf(1.0f - kxs*kxs - ky*ky, 0.0f));
      const float knorm = kzs*ct - kxs*st;
      float sr, cr;
      sincosf(ap * kzs, &sr, &cr);
      const float wj = kzs / (knorm + 1e-12f);
      if(hypotf(h_A[r*N+c].x, h_A[r*N+c].y) > 1e-2f*maxA){
        wjMin = fminf(wjMin, wj);
        wjMax = fmaxf(wjMax, wj);
      }
      h_C[r*N+c] = complexFormat(wj*(h_B[r*N+c].x*cr - h_B[r*N+c].y*sr),
                                 wj*(h_B[r*N+c].x*sr + h_B[r*N+c].y*cr));
    }
  }

  // ---------- Compare A vs C (Newton vs closed-form source points) ----------
  {
    float maxAbs = 0, maxRel = 0;
    for(int i = 0; i < N*N; i++){
      float d = hypotf(h_A[i].x-h_C[i].x, h_A[i].y-h_C[i].y);
      maxAbs = fmaxf(maxAbs, d);
      float aA = hypotf(h_A[i].x, h_A[i].y);
      if(aA > 1e-3f*maxA) maxRel = fmaxf(maxRel, d/aA);
    }
    fmt::println("A(gather) vs C(sine*wj*e^iDkz): maxAbsDiff={:.3e}  maxRelDiff(>0.1%% peak)={:.3e}", maxAbs, maxRel);
    fmt::println("Jacobian wj over beam (|A|>1%% peak): min={:.6f} max={:.6f}  (center=1/cosT={:.6f})", wjMin, wjMax, 1.0f/ct);
  }
  {
    float aAc = hypotf(h_A[N/2*N+N/2].x, h_A[N/2*N+N/2].y);
    float aBc = hypotf(h_B[N/2*N+N/2].x, h_B[N/2*N+N/2].y);
    fmt::println("k-space center: |A|={:.5f} |B|={:.5f}  |A|/|B|={:.6f} (expect 1/cosT={:.6f})", aAc, aBc, aAc/aBc, 1.0f/ct);
  }

  // ---------- IFFT both: real-space meaning of each ----------
  myCuDMalloc(complexFormat, d_kA, N*N);
  myCuDMalloc(complexFormat, d_kB, N*N);
  myCuDMalloc(complexFormat, d_sA, N*N);
  myCuDMalloc(complexFormat, d_sB, N*N);
  myMemcpyH2D(d_kA, h_A, N*N*sizeof(complexFormat));
  myMemcpyH2D(d_kB, h_B, N*N*sizeof(complexFormat));
  cudaConvertFO(d_kA);
  cudaConvertFO(d_kB);
  myIFFT(d_kA, d_sA);
  myIFFT(d_kB, d_sB);

  plt.plotComplex(d_sA, MOD, 0, 1, "step4_surf_A_gather.png", 1, 0, 1);
  plt.plotComplex(d_sB, MOD, 0, 1, "step4_surf_B_sine.png", 1, 0, 1);

  fmt::println("real-space IFFT:");
  reportWidth(d_sA, N, "A  (gather:  wj*e^{iDkz})");
  reportWidth(d_sB, N, "B  (sine kernel: no weight/phase)");
  fmt::println("B is the oblique-plane field at D=0 (pupil pattern sliced obliquely):");
  fmt::println("  expect FWHM_x~{:.2f}px FWHM_y~{:.2f}px", 2.0f*sqrtf(log(2.0f))*w0/ct, 2.0f*sqrtf(log(2.0f))*w0);
  {
    float wD = w0*sqrtf(1.0f + powf(2.0f*lambda*D/(M_PI*w0*pixelsize*w0*pixelsize), 2));
    fmt::println("A is the propagated beam at D: expect FWHM_x~{:.2f}px FWHM_y~{:.2f}px",
                 2.0f*sqrtf(log(2.0f))*wD/ct*1.0f, 2.0f*sqrtf(log(2.0f))*wD);
  }
  fmt::println("Done. step4_surf_A_gather.png / step4_surf_B_sine.png written to CWD.");
  return 0;
}
