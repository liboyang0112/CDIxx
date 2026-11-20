#include "cudaDefs_h.cu"
#include "cudaConfig.hpp"

// Free-space tilt correction around y-axis.
// Detector: tan(alpha) = (c-col/2)*a, a = pixelSize/distance
// Unit vector: (hx/s, hy/s, 1/s), s = sqrt(1+hx^2+hy^2)
// Scattering vector (sample frame): Q_x = (cos_t*hx + sin_t)/s - sin_t, Q_y = hy/s
// At detector center (hx=0): Q_x = 0 (specular direction)
// Uses Catmull-Rom cubic interpolation (4x4 neighborhood) to preserve high frequencies.
// beamstop: Real** — if NULL, allocated and filled with 1. OOB pixels set to 0.

// Simulate (grid->detector): for each detector pixel, compute grid position (explicit)
cuFunc(reflSimulateKernel, (Real* input, Real* output,
                             float sin_t, float cos_t, float a),
       (input, output, sin_t, cos_t, a), {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index >= cuda_row*cuda_column) return;
  int r = index / cuda_column;
  int c = index % cuda_column;

  float hx = (c - cuda_column/2.0f) * a;
  float hy = (r - cuda_row/2.0f) * a;
  float s = sqrtf(1.0f + hx*hx + hy*hy);

  float Qx = (cos_t*hx + sin_t) / s - sin_t;
  float Qy = hy / s;

  float mc = Qx / a + cuda_column/2.0f;
  float mr = Qy / a + cuda_row/2.0f;

  int r0 = (int)floorf(mr) - 1, c0 = (int)floorf(mc) - 1;
  float fr = mr - (float)((int)floorf(mr)), fc = mc - (float)((int)floorf(mc));

  if(r0 < 0 || r0+3 >= cuda_row || c0 < 0 || c0+3 >= cuda_column){
    output[index] = 0;
    return;
  }
  float wr0 = 0.5f*(-fr + 2*fr*fr - fr*fr*fr);
  float wr1 = 0.5f*(2 - 5*fr*fr + 3*fr*fr*fr);
  float wr2 = 0.5f*(fr + 4*fr*fr - 3*fr*fr*fr);
  float wr3 = 0.5f*(-fr*fr + fr*fr*fr);
  float wc0 = 0.5f*(-fc + 2*fc*fc - fc*fc*fc);
  float wc1 = 0.5f*(2 - 5*fc*fc + 3*fc*fc*fc);
  float wc2 = 0.5f*(fc + 4*fc*fc - 3*fc*fc*fc);
  float wc3 = 0.5f*(-fc*fc + fc*fc*fc);

  float sum = 0;
  sum += wr0*(wc0*input[(r0+0)*cuda_column+c0+0] + wc1*input[(r0+0)*cuda_column+c0+1] + wc2*input[(r0+0)*cuda_column+c0+2] + wc3*input[(r0+0)*cuda_column+c0+3]);
  sum += wr1*(wc0*input[(r0+1)*cuda_column+c0+0] + wc1*input[(r0+1)*cuda_column+c0+1] + wc2*input[(r0+1)*cuda_column+c0+2] + wc3*input[(r0+1)*cuda_column+c0+3]);
  sum += wr2*(wc0*input[(r0+2)*cuda_column+c0+0] + wc1*input[(r0+2)*cuda_column+c0+1] + wc2*input[(r0+2)*cuda_column+c0+2] + wc3*input[(r0+2)*cuda_column+c0+3]);
  sum += wr3*(wc0*input[(r0+3)*cuda_column+c0+0] + wc1*input[(r0+3)*cuda_column+c0+1] + wc2*input[(r0+3)*cuda_column+c0+2] + wc3*input[(r0+3)*cuda_column+c0+3]);
  output[index] = sum;
})

// Correct (detector->grid): for each grid pixel, solve for detector position via iteration
cuFunc(reflCorrectKernel, (Real* input, Real* output, Real* beamstop,
                            float sin_t, float cos_t, float a),
       (input, output, beamstop, sin_t, cos_t, a), {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index >= cuda_row*cuda_column) return;
  int r = index / cuda_column;
  int c = index % cuda_column;

  float Qx = (c - cuda_column/2.0f) * a;
  float Qy = (r - cuda_row/2.0f) * a;

  // Solve: Qx = (cos_t*hx + sin_t)/s - sin_t, Qy = hy/s, s = sqrt(1+hx^2+hy^2)
  // Rearranged: (Qx + sin_t)*s = cos_t*hx + sin_t
  // Fixed-point: hx = ((Qx+sin_t)*s - sin_t)/cos_t, hy = Qy*s
  float hx = Qx / cos_t, hy = Qy;
  // Iteration 1
  float s = sqrtf(1.0f + hx*hx + hy*hy);
  hx = ((Qx + sin_t) * s - sin_t) / cos_t;
  hy = Qy * s;
  // Iteration 2
  s = sqrtf(1.0f + hx*hx + hy*hy);
  hx = ((Qx + sin_t) * s - sin_t) / cos_t;
  hy = Qy * s;
  // Iteration 3
  s = sqrtf(1.0f + hx*hx + hy*hy);
  hx = ((Qx + sin_t) * s - sin_t) / cos_t;
  hy = Qy * s;

  float mc = hx / a + cuda_column/2.0f;
  float mr = hy / a + cuda_row/2.0f;

  int r0 = (int)floorf(mr) - 1, c0 = (int)floorf(mc) - 1;
  float fr = mr - (float)((int)floorf(mr)), fc = mc - (float)((int)floorf(mc));

  if(r0 < 0 || r0+3 >= cuda_row || c0 < 0 || c0+3 >= cuda_column){
    output[index] = 0;
    beamstop[index] = 1;
    return;
  }
  float wr0 = 0.5f*(-fr + 2*fr*fr - fr*fr*fr);
  float wr1 = 0.5f*(2 - 5*fr*fr + 3*fr*fr*fr);
  float wr2 = 0.5f*(fr + 4*fr*fr - 3*fr*fr*fr);
  float wr3 = 0.5f*(-fr*fr + fr*fr*fr);
  float wc0 = 0.5f*(-fc + 2*fc*fc - fc*fc*fc);
  float wc1 = 0.5f*(2 - 5*fc*fc + 3*fc*fc*fc);
  float wc2 = 0.5f*(fc + 4*fc*fc - 3*fc*fc*fc);
  float wc3 = 0.5f*(-fc*fc + fc*fc*fc);

  float sum = 0;
  sum += wr0*(wc0*input[(r0+0)*cuda_column+c0+0] + wc1*input[(r0+0)*cuda_column+c0+1] + wc2*input[(r0+0)*cuda_column+c0+2] + wc3*input[(r0+0)*cuda_column+c0+3]);
  sum += wr1*(wc0*input[(r0+1)*cuda_column+c0+0] + wc1*input[(r0+1)*cuda_column+c0+1] + wc2*input[(r0+1)*cuda_column+c0+2] + wc3*input[(r0+1)*cuda_column+c0+3]);
  sum += wr2*(wc0*input[(r0+2)*cuda_column+c0+0] + wc1*input[(r0+2)*cuda_column+c0+1] + wc2*input[(r0+2)*cuda_column+c0+2] + wc3*input[(r0+2)*cuda_column+c0+3]);
  sum += wr3*(wc0*input[(r0+3)*cuda_column+c0+0] + wc1*input[(r0+3)*cuda_column+c0+1] + wc2*input[(r0+3)*cuda_column+c0+2] + wc3*input[(r0+3)*cuda_column+c0+3]);
  output[index] = sum;
})

// === Sine lens model (no Jacobian weight, u = sin(alpha) directly) ===
// Formula: gx = cos(theta)*u + sin(theta)*(1-w), w = sqrt(1-u^2-v^2)
// y is identity (tilt around y-axis). No weight.
// Uses Catmull-Rom cubic interpolation in x (4 samples), row is passthrough.

// Correct (detector->grid): for each grid pixel, solve for detector u via Newton.
cuFunc(reflCorrectSineKernel, (Real* input, Real* output, Real* beamstop,
                                float sin_t, float cos_t, float a),
       (input, output, beamstop, sin_t, cos_t, a), {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index >= cuda_row*cuda_column) return;
  int r = index / cuda_column;
  int c = index % cuda_column;

  float target = (c - cuda_column/2.0f) * a;
  float v = (r - cuda_row/2.0f) * a;

  // Solve: target = cos_t*u + sin_t*(1-sqrt(1-u^2-v^2)) for u
  float u = target / cos_t;
  // Newton iteration 1
  float w = sqrtf(1.0f - u*u - v*v);
  if(w < 1e-6f){ output[index] = 0; beamstop[index] = 0; return; }
  float f = cos_t*u + sin_t*(1.0f-w) - target;
  float fp = cos_t + sin_t*u/w;
  u -= f/fp;
  // Newton iteration 2
  w = sqrtf(1.0f - u*u - v*v);
  if(w < 1e-6f){ output[index] = 0; beamstop[index] = 0; return; }
  f = cos_t*u + sin_t*(1.0f-w) - target;
  fp = cos_t + sin_t*u/w;
  u -= f/fp;

  float mc = u/a + cuda_column/2.0f;

  int c0 = (int)floorf(mc) - 1;
  float fc = mc - (float)((int)floorf(mc));

  if(c0 < 0 || c0+3 >= cuda_column){
    output[index] = 0;
    beamstop[index] = 1;
    return;
  }
  float wc0 = 0.5f*(-fc + 2*fc*fc - fc*fc*fc);
  float wc1 = 0.5f*(2 - 5*fc*fc + 3*fc*fc*fc);
  float wc2 = 0.5f*(fc + 4*fc*fc - 3*fc*fc*fc);
  float wc3 = 0.5f*(-fc*fc + fc*fc*fc);

  output[index] = wc0*input[r*cuda_column+c0+0] + wc1*input[r*cuda_column+c0+1]
                + wc2*input[r*cuda_column+c0+2] + wc3*input[r*cuda_column+c0+3];
})

// Simulate (grid->detector): for each detector pixel, compute grid position (explicit).
cuFunc(reflSimulateSineKernel, (Real* input, Real* output,
                                 float sin_t, float cos_t, float a),
       (input, output, sin_t, cos_t, a), {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index >= cuda_row*cuda_column) return;
  int r = index / cuda_column;
  int c = index % cuda_column;

  float u = (c - cuda_column/2.0f) * a;
  float v = (r - cuda_row/2.0f) * a;
  float w2 = 1.0f - u*u - v*v;
  if(w2 <= 0.0f){ output[index] = 0; return; }
  float w = sqrtf(w2);

  // Grid position: x_grid = (cos_t*u + sin_t*(1-w)) / a
  float mc = (cos_t*u + sin_t*(1.0f-w)) / a + cuda_column/2.0f;

  int c0 = (int)floorf(mc) - 1;
  float fc = mc - (float)((int)floorf(mc));

  if(c0 < 0 || c0+3 >= cuda_column){
    output[index] = 0;
    return;
  }
  float wc0 = 0.5f*(-fc + 2*fc*fc - fc*fc*fc);
  float wc1 = 0.5f*(2 - 5*fc*fc + 3*fc*fc*fc);
  float wc2 = 0.5f*(fc + 4*fc*fc - 3*fc*fc*fc);
  float wc3 = 0.5f*(-fc*fc + fc*fc*fc);

  output[index] = wc0*input[r*cuda_column+c0+0] + wc1*input[r*cuda_column+c0+1]
                + wc2*input[r*cuda_column+c0+2] + wc3*input[r*cuda_column+c0+3];
})

// === Ewald sphere curvature correction (no tilt) ===
// correctToRegular: detector (curved) -> regular grid, with s^2 Jacobian weight
cuFunc(ewaldCorrectKernel, (Real* input, Real* output, Real* beamstop, float a, float centerPx, float centerPy), (input, output, beamstop, a, centerPx, centerPy), {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index >= cuda_row*cuda_column) return;
  int r = index / cuda_column;
  int c = index % cuda_column;

  float xi = (c - cuda_column/2.0f) * a;
  float eta = (r - cuda_row/2.0f) * a;
  float q2 = xi*xi + eta*eta;
  if(q2 >= 1.0f) { output[index] = 0; beamstop[index] = 0; return; }

  float s_inv = sqrtf(1.0f - q2);
  float mr = (r - cuda_row/2.0f) / s_inv + centerPy;
  float mc = (c - cuda_column/2.0f) / s_inv + centerPx;
  float weight = 1.0f / (1.0f - q2);

  int r0 = (int)floorf(mr) - 1, c0 = (int)floorf(mc) - 1;
  float fr = mr - (float)((int)floorf(mr)), fc = mc - (float)((int)floorf(mc));

  if(r0 < 0 || r0+3 >= cuda_row || c0 < 0 || c0+3 >= cuda_column){
    output[index] = 0;
    beamstop[index] = 1;
    return;
  }
  float wr0 = 0.5f*(-fr + 2*fr*fr - fr*fr*fr);
  float wr1 = 0.5f*(2 - 5*fr*fr + 3*fr*fr*fr);
  float wr2 = 0.5f*(fr + 4*fr*fr - 3*fr*fr*fr);
  float wr3 = 0.5f*(-fr*fr + fr*fr*fr);
  float wc0 = 0.5f*(-fc + 2*fc*fc - fc*fc*fc);
  float wc1 = 0.5f*(2 - 5*fc*fc + 3*fc*fc*fc);
  float wc2 = 0.5f*(fc + 4*fc*fc - 3*fc*fc*fc);
  float wc3 = 0.5f*(-fc*fc + fc*fc*fc);

  float sum = 0;
  sum += wr0*(wc0*input[(r0+0)*cuda_column+c0+0] + wc1*input[(r0+0)*cuda_column+c0+1] + wc2*input[(r0+0)*cuda_column+c0+2] + wc3*input[(r0+0)*cuda_column+c0+3]);
  sum += wr1*(wc0*input[(r0+1)*cuda_column+c0+0] + wc1*input[(r0+1)*cuda_column+c0+1] + wc2*input[(r0+1)*cuda_column+c0+2] + wc3*input[(r0+1)*cuda_column+c0+3]);
  sum += wr2*(wc0*input[(r0+2)*cuda_column+c0+0] + wc1*input[(r0+2)*cuda_column+c0+1] + wc2*input[(r0+2)*cuda_column+c0+2] + wc3*input[(r0+2)*cuda_column+c0+3]);
  sum += wr3*(wc0*input[(r0+3)*cuda_column+c0+0] + wc1*input[(r0+3)*cuda_column+c0+1] + wc2*input[(r0+3)*cuda_column+c0+2] + wc3*input[(r0+3)*cuda_column+c0+3]);
  output[index] = weight * sum;
})

// simulateTilted: regular grid -> detector (curved), with s^2 Jacobian weight
cuFunc(ewaldSimulateKernel, (Real* input, Real* output, float a, float centerPx, float centerPy), (input, output, a, centerPx, centerPy), {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index >= cuda_row*cuda_column) return;
  int r = index / cuda_column;
  int c = index % cuda_column;

  float u = (c - centerPx) * a;
  float v = (r - centerPy) * a;
  float weight = 1.0f + u*u + v*v;
  float s = sqrtf(weight);
  float mr = (r - centerPy) / s + cuda_row/2.0f;
  float mc = (c - centerPx) / s + cuda_column/2.0f;

  int r0 = (int)floorf(mr) - 1, c0 = (int)floorf(mc) - 1;
  float fr = mr - (float)((int)floorf(mr)), fc = mc - (float)((int)floorf(mc));

  if(r0 < 0 || r0+3 >= cuda_row || c0 < 0 || c0+3 >= cuda_column){
    output[index] = 0;
    return;
  }
  float wr0 = 0.5f*(-fr + 2*fr*fr - fr*fr*fr);
  float wr1 = 0.5f*(2 - 5*fr*fr + 3*fr*fr*fr);
  float wr2 = 0.5f*(fr + 4*fr*fr - 3*fr*fr*fr);
  float wr3 = 0.5f*(-fr*fr + fr*fr*fr);
  float wc0 = 0.5f*(-fc + 2*fc*fc - fc*fc*fc);
  float wc1 = 0.5f*(2 - 5*fc*fc + 3*fc*fc*fc);
  float wc2 = 0.5f*(fc + 4*fc*fc - 3*fc*fc*fc);
  float wc3 = 0.5f*(-fc*fc + fc*fc*fc);

  float sum = 0;
  sum += wr0*(wc0*input[(r0+0)*cuda_column+c0+0] + wc1*input[(r0+0)*cuda_column+c0+1] + wc2*input[(r0+0)*cuda_column+c0+2] + wc3*input[(r0+0)*cuda_column+c0+3]);
  sum += wr1*(wc0*input[(r0+1)*cuda_column+c0+0] + wc1*input[(r0+1)*cuda_column+c0+1] + wc2*input[(r0+1)*cuda_column+c0+2] + wc3*input[(r0+1)*cuda_column+c0+3]);
  sum += wr2*(wc0*input[(r0+2)*cuda_column+c0+0] + wc1*input[(r0+2)*cuda_column+c0+1] + wc2*input[(r0+2)*cuda_column+c0+2] + wc3*input[(r0+2)*cuda_column+c0+3]);
  sum += wr3*(wc0*input[(r0+3)*cuda_column+c0+0] + wc1*input[(r0+3)*cuda_column+c0+1] + wc2*input[(r0+3)*cuda_column+c0+2] + wc3*input[(r0+3)*cuda_column+c0+3]);
  output[index] = weight * sum;
})
