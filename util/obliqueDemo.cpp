#include "fmt/core.h"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "propagator.hpp"
#include "misc.hpp"
#include <math.h>
using namespace std;

int main(){
  int N = 1024;
  Real pixelsize = 2.0;
  Real lambda = 0.6328;
  Real D = 3e3;  // past Rayleigh distance for visible fringes
  //Real theta = M_PI/4;  // 45 degrees
  Real theta = 0.1;  // 45 degrees

  init_cuda_image();
  resize_cuda_image(N, N);
  plt.init(N, N);
  init_fft(N, N);

  // Create Gaussian beam at pupil plane
  myCuDMallocClean(complexFormat, d_pupil, N*N);

  // Build Gaussian on GPU via host (moderate waist for visible fringes)
  myDMalloc(complexFormat, h_pupil, N*N);
  float w0 = 128.0;  // beam waist in pixels (wider angular spectrum for visible distortion)
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

  // Propagator setup
  propagator prop;
  prop.lambda = lambda;
  prop.distance = D;
  prop.pixelsize = pixelsize;
  prop.row = N;
  prop.column = N;

  // 1. Standard propagation (straight) — circular wavefront
  myCuDMalloc(complexFormat, d_straight, N*N);
  prop.angularSpectrumPropagate(d_pupil, d_straight);
  plt.plotComplexColor(d_straight, 0, 1, "straight_color");

  // 2. Oblique propagation (k-vector rotation into surface frame)
  myCuDMalloc(complexFormat, d_oblique, N*N);
  prop.obliquePropagateFwd(d_pupil, d_oblique, theta);

  plt.plotComplexColor(d_oblique, 0, 1, "oblique_color");

  // 3. Intensity (amplitude only) — should be smooth Gaussian, no strips
  plt.plotComplex(d_oblique, MOD, 0, 1, "oblique_intensity.png", 1, 0, 1);
  plt.plotComplex(d_straight, MOD, 0, 1, "straight_intensity.png", 1, 0, 1);

  fmt::println("Done. Check output:");
  fmt::println("  straight_color.jpg  - complex color (circular wavefront)");
  fmt::println("  oblique_color.jpg   - complex color (elliptical via k-vector rotation)");

  return 0;
}
