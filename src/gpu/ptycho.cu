#include "cudaDefs_h.cu"
#include <curand_kernel.h>

#define ALPHA 0.5
#define BETA 0.5
#define DELTA 1e-3
#define GAMMA 0.5

__forceinline__ __device__ __host__ Real gaussian(Real x, Real y, Real sigma){
  return exp(-(x*x+y*y)/2/(sigma*sigma));
}

cuFunc(applySupport,(Real* image, Real* support),(image,support),{
  cuda1Idx();
  if(support[index] > vars->threshold) image[index] = 0;
})
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

__forceinline__ __device__ void rPIE(cuComplex &target, cuComplex source, cuComplex &diff, Real maxi, Real param){
  Real denom = source.x*source.x+source.y*source.y;
//  if(denom < 8e-4*maxi) return;
  denom = 1./((1-param)*denom+param*maxi);
  target.x -= (source.x*diff.x + source.y*diff.y)*denom;
  target.y -= (source.x*diff.y - source.y*diff.x)*denom;
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
  rPIE(objectdat, probedat, diff, mod2maxProbe, ALPHA);
  rPIE(probedat, objectdat, diff, mod2maxObj, BETA);
  probe[index] = probedat;
  object[index] = objectdat;
})

cuFuncc(random,(complexFormat* object, void *state),(cuComplex* object, curandStateMRG32k3a *state),((cuComplex*)object, (curandStateMRG32k3a*)state),{
  cuda1Idx()
  curand_init(1,index,0,state+index);
  object[index].x = curand_uniform(&state[index]);
  object[index].y = curand_uniform(&state[index]);
})

cuFuncc(pupilFunc,(complexFormat* object),(cuComplex* object),((cuComplex*)object),{
  cudaIdx()
  int shiftx = x - cuda_row/2;
  int shifty = y - cuda_column/2;
  object[index].x = 3*gaussian(shiftx,shifty,cuda_row/8);
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
  if(beamstop[index] > 0.5){
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
       (complexFormat* d_fft_data, int width, int height),
       (cuComplex* d_fft_data, int width, int height),
       ((cuComplex*)d_fft_data, width, height),
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;  // kx: 0 to width/2
    int ky = blockIdx.y * blockDim.y + threadIdx.y;  // ky: 0 to height-1

    int kx_max = width / 2 + 1;  // number of stored kx per row
    if (kx >= kx_max || ky >= height) return;

    // Map kx back to physical frequency: [0..w/2] corresponds to positive frequencies
    // Negative kx are implicit via conjugate symmetry
    int kx_phys = kx;
    int ky_phys = (ky <= height/2) ? ky : ky - height;

    // Avoid DC component
    if (kx_phys == 0 && ky_phys == 0) {
        d_fft_data[ky * kx_max + kx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    // Compute discrete Laplacian denominator: 4*sin²(π kx/N) + 4*sin²(π ky/M)
    Real fx = sinf(M_PI * (Real)kx_phys / (Real)width);
    Real fy = sinf(M_PI * (Real)abs(ky_phys) / (Real)height);  // symmetric
    Real denom = 4.0f * (fx*fx + fy*fy);

    int idx = ky * kx_max + kx;
    cuComplex& val = d_fft_data[idx];
    Real scale = 1.0f / denom;

    val.x *= scale;
    val.y *= scale;
})
cuFunc(phaseUnwrapping, 
       (Real* d_wrapped_phase, Real* d_unwrapped_phase, int width, int height), 
       (d_wrapped_phase, d_unwrapped_phase, width, height),
{
    // Temporary buffers (assumed pre-allocated or managed externally)
    extern __shared__ char shared_mem[];
    Real* d_gx = (Real*)shared_mem;                        // gradient x
    Real* d_gy = (Real*)(d_gx + width * height);           // gradient y
    Real* d_b  = (Real*)(d_gy + width * height);           // divergence (RHS of Poisson)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % width;
    int y = idx / width;

    if (idx >= width * height) return;

    // === Step 1: Compute wrapped gradients with jump correction ===
    Real right = (x + 1 < width) ? d_wrapped_phase[idx + 1] : d_wrapped_phase[idx];
    Real down  = (y + 1 < height) ? d_wrapped_phase[idx + width] : d_wrapped_phase[idx];

    // Forward differences
    Real dx = right - d_wrapped_phase[idx];
    Real dy = down  - d_wrapped_phase[idx];

    // Remove 2π discontinuities
    dx += round((M_PI - dx) / (2.0 * M_PI)) * 2.0 * M_PI;
    dy += round((M_PI - dy) / (2.0 * M_PI)) * 2.0 * M_PI;

    d_gx[idx] = dx;
    d_gy[idx] = dy;

    __syncthreads();
    // === Step 2: Compute divergence b = ∂²φ/∂x² + ∂²φ/∂y² ≈ div(grad φ) ===
    Real lapx = 0, lapy = 0;
    if (x > 0 && x < width - 1) {
        Real left_val = d_gx[idx - 1];
        Real right_val = d_gx[idx + 1];
        lapx = 0.5f * (right_val - left_val);  // central diff of gx
    } else {
        lapx = (x == 0) ? (d_gx[idx+1] - d_gx[idx]) : (d_gx[idx] - d_gx[idx-1]);
    }
    if (y > 0 && y < height - 1) {
        Real up_val   = d_gy[idx - width];
        Real down_val = d_gy[idx + width];
        lapy = 0.5f * (down_val - up_val);
    } else {
        lapy = (y == 0) ? (d_gy[idx+width] - d_gy[idx]) : (d_gy[idx] - d_gy[idx-width]);
    }
    d_b[idx] = lapx + lapy;  // this is the "residue density" or source term
})


