#include "cudaDefs_h.cu"
#include "cuComplex.h"
#include <curand_kernel.h>

#define ALPHA 0.5
#define BETA 0.5
#define DELTA 1e-3

cuFuncc(multiplyProbe,(complexFormat* object, complexFormat* probe, complexFormat* U, int shiftx, int shifty, int objrow, int objcol, complexFormat *window = 0),(cuComplex* object, cuComplex* probe, cuComplex* U, int shiftx, int shifty, int objrow, int objcol, cuComplex* window),((cuComplex*)object,(cuComplex*)probe,(cuComplex*)U,shiftx,shifty,objrow,objcol,(cuComplex*)window),{
  cudaIdx();
  cuComplex tmp;
  if(x+shiftx >= objrow || y+shifty >= objcol || x+shiftx < 0 || y+shifty < 0) tmp.x = tmp.y = 0;
  else tmp =  object[(x+shiftx)*objcol+y+shifty];
  if(window) window[index] = tmp;
  U[index] = cuCmulf(probe[index], tmp);
})

cuFuncc(getWindow,(complexFormat* object, int shiftx, int shifty, int objrow, int objcol, complexFormat *window),(cuComplex* object, int shiftx, int shifty, int objrow, int objcol, cuComplex* window),((cuComplex*)object,shiftx,shifty,objrow,objcol,(cuComplex*)window),{
  cudaIdx();
  cuComplex tmp;
  if(x+shiftx >= objrow || y+shifty >= objcol || x+shiftx < 0 || y+shifty < 0) tmp.x = tmp.y = 0;
  else tmp =  object[(x+shiftx)*objcol+y+shifty];
  window[index] = tmp;
})

cuFuncc(updateWindow,(complexFormat* object, int shiftx, int shifty, int objrow, int objcol, complexFormat *window),(cuComplex* object, int shiftx, int shifty, int objrow, int objcol, cuComplex* window),((cuComplex*)object,shiftx,shifty,objrow,objcol,(cuComplex*)window),{
  cudaIdx();
  if(x+shiftx >= objrow || y+shifty >= objcol || x+shiftx < 0 || y+shifty < 0) return;
  object[(x+shiftx)*objcol+y+shifty] = window[index];
})


__forceinline__ __device__ void ePIE(cuComplex &target, cuComplex source, cuComplex &diff, Real maxi, Real param){
  Real denom = param/(maxi);
  source = cuCmulf(cuConjf(source),diff);
  target.x -= source.x*denom;
  target.y -= source.y*denom;
}

__forceinline__ __device__ void rPIE(cuComplex &target, cuComplex source, cuComplex &diff, Real maxi, Real param, Real stepsize = 1){
  Real denom = source.x*source.x+source.y*source.y;
//  if(denom < 8e-4*maxi) return;
  denom = stepsize/((1-param)*denom+param*maxi);
  target.x -= (source.x*diff.x + source.y*diff.y)*denom;
  target.y -= (source.x*diff.y - source.y*diff.x)*denom;
}
__forceinline__ __device__ void rPIE_step(cuComplex &target, cuComplex source, cuComplex &diff, Real maxi, Real param){
  Real denom = source.x*source.x+source.y*source.y;
//  if(denom < 8e-4*maxi) return;
  denom = 1./((1-param)*denom+param*maxi);
  target.x =-(source.x*diff.x + source.y*diff.y)*denom;
  target.y =-(source.x*diff.y - source.y*diff.x)*denom;
}

cuFuncc(updateObject,(complexFormat* object, complexFormat* probe, complexFormat* U, Real mod2maxProbe),(cuComplex* object, cuComplex* probe, cuComplex* U, Real mod2maxProbe),((cuComplex*)object,(cuComplex*)probe,(cuComplex*)U,mod2maxProbe),{
  cuda1Idx()
  rPIE(object[index], probe[index], U[index], mod2maxProbe, ALPHA);
  //ePIE(object[index], probe[index], U[index], mod2maxProbe, ALPHA);
})

cuFuncc(updateObjectAndProbe,(complexFormat* object, complexFormat* probe, complexFormat* U, Real mod2maxProbe, Real mod2maxObj),(cuComplex* object, cuComplex* probe, cuComplex* U, Real mod2maxProbe, Real mod2maxObj),((cuComplex*)object,(cuComplex*)probe,(cuComplex*)U,mod2maxProbe,mod2maxObj),{
  cuda1Idx()
  cuComplex objectdat= object[index];
  cuComplex probedat= probe[index];
  cuComplex diff= U[index];
  //ePIE(object[index], probe[index], diff, mod2maxProbe, ALPHA);
  //ePIE(probe[index], objectdat, diff, mod2maxObj, BETA);
  rPIE(objectdat, probedat, diff, mod2maxProbe, ALPHA, 0.5);
  rPIE(probedat, objectdat, diff, mod2maxObj, BETA,0.5);
  probe[index] = probedat;
  object[index] = objectdat;
})

cuFuncc(updateObjectAndProbeStep,(complexFormat* object, complexFormat* probe, complexFormat* probeStep, complexFormat* U, Real mod2maxProbe, Real mod2maxObj),(cuComplex* object, cuComplex* probe, cuComplex* probeStep, cuComplex* U, Real mod2maxProbe, Real mod2maxObj),((cuComplex*)object,(cuComplex*)probe,(cuComplex*)probeStep,(cuComplex*)U,mod2maxProbe,mod2maxObj),{
  cuda1Idx()
  cuComplex objectdat= object[index];
  cuComplex probedat= probe[index];
  cuComplex diff= U[index];
  //ePIE(object[index], probe[index], diff, mod2maxProbe, ALPHA);
  //ePIE(probe[index], objectdat, diff, mod2maxObj, BETA);
  rPIE(objectdat, probedat, diff, mod2maxProbe, ALPHA);
  rPIE_step(probedat, objectdat, diff, mod2maxObj, BETA);
  probeStep[index] = cuCaddf(probeStep[index], probedat);
  object[index] = objectdat;
})

cuFuncc(random,(complexFormat* object, void *state),(cuComplex* object, curandStateMRG32k3a *state),((cuComplex*)object, (curandStateMRG32k3a*)state),{
  cuda1Idx()
  Real phaseshift = curand_uniform(&state[index]);
  Real c, s;
  sincosf(phaseshift*2*M_PI, &s, &c);
  object[index].x = c;
  object[index].y = s;
})

__device__ Real gaussian(float x, float y, float sigma){
  return exp(-(x*x+y*y)/2/(sigma*sigma));
}
cuFuncc(pupilFunc,(complexFormat* object),(cuComplex* object),((cuComplex*)object),{
  cudaIdx()
  int shiftx = x - cuda_row/2;
  int shifty = y - cuda_column/2;
  object[index].x = 3*gaussian(shiftx,shifty,cuda_row/4);
  object[index].y = 0;
})

cuFuncc(multiplyx,
    (complexFormat* object, complexFormat* out),
    (cuComplex* object, cuComplex* out),
    ((cuComplex*)object, (cuComplex*)out),
{
    cuda1Idx();
    Real x = ((unsigned int)index) / (unsigned int)cuda_column;  // Encourage uint div
    Real scale = x * (1.0f / (Real)cuda_row) - 0.5f;
    out[index].x = object[index].x * scale;
    out[index].y = object[index].y * scale;
})


cuFuncc(multiplyy,
    (complexFormat* object, complexFormat* out),
    (cuComplex* object, cuComplex* out),
    ((cuComplex*)object, (cuComplex*)out),
{
    cuda1Idx();
    Real y = index % cuda_column;
    Real scale = y * (1.0f / (Real)cuda_column) - 0.5f;
    out[index].x = object[index].x * scale;
    out[index].y = object[index].y * scale;
})

cuFuncc(calcPartial,(Real* out, complexFormat* object, complexFormat* Fn, Real* pattern, Real* beamstop),(Real* out, cuComplex* object, cuComplex* Fn, Real* pattern, Real* beamstop),(out, (cuComplex*)object,(cuComplex*)Fn,pattern,beamstop),{
  cuda1Idx();
  if(beamstop && beamstop[index] > 0.5){
    object[index].x = 0;
    return;
  }
  auto fntmp = Fn[index];
  Real fnmod2 = fntmp.x*fntmp.x + fntmp.y*fntmp.y;
  Real ret = fntmp.x*object[index].y - fntmp.y*object[index].x;
  Real fact = pattern[index]+DELTA;
  if(fact<0) fact = 0;
  ret*=1-sqrtf(fact/(fnmod2+DELTA));
  out[index] = ret;
})

cuFuncc(solve_poisson_frequency_domain,
       (complexFormat* d_fft_data),
       (cuComplex* d_fft_data),
       ((cuComplex*)d_fft_data),
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // x: 0 to cuda_row/2
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // y: 0 to cuda_column-1

    int x_max = cuda_row / 2 + 1;  // number of stored x per row
    if (x >= x_max || y >= cuda_column) return;

    // Map x back to physical frequency: [0..w/2] corresponds to positive frequencies
    // Negative x are implicit via conjugate symmetry
    int x_phys = x;
    int y_phys = (y <= cuda_column/2) ? y : y - cuda_column;

    // Avoid DC component
    if (x_phys == 0 && y_phys == 0) {
        d_fft_data[y * x_max + x] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    // Compute discrete Laplacian denominator: 4*sin²(π x/N) + 4*sin²(π y/M)
    Real fx = sinf(M_PI * (Real)x_phys / (Real)((x_max-1)*2));
    Real fy = sinf(M_PI * (Real)abs(y_phys) / (Real)cuda_column);  // symmetric
    Real denom = 4.0f * (fx*fx + fy*fy);

    int idx = y * x_max + x;
    cuComplex& val = d_fft_data[idx];
    Real scale = 1.0f / denom;

    val.x *= scale;
    val.y *= scale;
})
cuFunc(phaseUnwrapping, 
       (Real* d_wrapped_phase), 
       (d_wrapped_phase),
{
    // Temporary buffers (assumed pre-allocated or managed externally)
    cudaIdx()
    extern __shared__ char shared_mem[];
    Real* d_gx = (Real*)shared_mem;                        // gradient x
    Real* d_gy = (Real*)(d_gx + cuda_row * cuda_column);           // gradient y
    Real* d_b  = (Real*)(d_gy + cuda_row * cuda_column);           // divergence (RHS of Poisson)


    Real right = (x + 1 < cuda_row) ? d_wrapped_phase[index + 1] : d_wrapped_phase[index];
    Real down  = (y + 1 < cuda_column) ? d_wrapped_phase[index + cuda_row] : d_wrapped_phase[index];

    // Forward differences
    Real dx = right - d_wrapped_phase[index];
    Real dy = down  - d_wrapped_phase[index];

    // Remove 2π discontinuities
    dx += round((M_PI - dx) / (2.0 * M_PI)) * 2.0 * M_PI;
    dy += round((M_PI - dy) / (2.0 * M_PI)) * 2.0 * M_PI;

    d_gx[index] = dx;
    d_gy[index] = dy;

    __syncthreads();
    // === Step 2: Compute divergence b = ∂²φ/∂x² + ∂²φ/∂y² ≈ div(grad φ) ===
    Real lapx = 0, lapy = 0;
    if (x > 0 && x < cuda_row - 1) {
        Real left_val = d_gx[index - 1];
        Real right_val = d_gx[index + 1];
        lapx = 0.5f * (right_val - left_val);  // central diff of gx
    } else {
        lapx = (x == 0) ? (d_gx[index+1] - d_gx[index]) : (d_gx[index] - d_gx[index-1]);
    }
    if (y > 0 && y < cuda_column - 1) {
        Real up_val   = d_gy[index - cuda_row];
        Real down_val = d_gy[index + cuda_row];
        lapy = 0.5f * (down_val - up_val);
    } else {
        lapy = (y == 0) ? (d_gy[index+cuda_row] - d_gy[index]) : (d_gy[index] - d_gy[index-cuda_row]);
    }
    d_b[index] = lapx + lapy;  // this is the "residue density" or source term
})


