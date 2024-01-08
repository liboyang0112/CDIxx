#include "cudaDefs_h.cu"
#include <curand_kernel.h>

#define ALPHA 0.5
#define BETA 1
#define DELTA 1e-3
#define GAMMA 0.5

__device__ __host__ Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
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


__device__ void ePIE(cuComplex &target, cuComplex source, cuComplex &diff, Real maxi, Real param){
  Real denom = param/(maxi);
  source = cuCmulf(cuConjf(source),diff);
  target.x -= source.x*denom;
  target.y -= source.y*denom;
}

__device__ void rPIE(cuComplex &target, cuComplex source, cuComplex &diff, Real maxi, Real param){
  Real denom = source.x*source.x+source.y*source.y;
//  if(denom < 8e-4*maxi) return;
  denom = 1./((1-param)*denom+param*maxi);
  source = cuCmulf(cuConjf(source),diff);
  target.x -= source.x*denom;
  target.y -= source.y*denom;
}

cuFuncc(updateObject,(complexFormat* object, complexFormat* probe, complexFormat* U, Real mod2maxProbe),(cuComplex* object, cuComplex* probe, cuComplex* U, Real mod2maxProbe),((cuComplex*)object,(cuComplex*)probe,(cuComplex*)U,mod2maxProbe),{
  cuda1Idx()
  rPIE(object[index], probe[index], U[index], mod2maxProbe, ALPHA);
})

cuFuncc(updateObjectAndProbe,(complexFormat* object, complexFormat* probe, complexFormat* U, Real mod2maxProbe, Real mod2maxObj),(cuComplex* object, cuComplex* probe, cuComplex* U, Real mod2maxProbe, Real mod2maxObj),((cuComplex*)object,(cuComplex*)probe,(cuComplex*)U,mod2maxProbe,mod2maxObj),{
  cuda1Idx()
  cuComplex objectdat= object[index];
  cuComplex diff= U[index];
  rPIE(object[index], probe[index], diff, mod2maxProbe, ALPHA);
  rPIE(probe[index], objectdat, diff, mod2maxObj, BETA);
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

cuFuncc(multiplyx,(complexFormat* object),(cuComplex* object),((cuComplex*)object),{
  cuda1Idx();
  int x = index/cuda_column;
  object[index].x *= Real(x)/cuda_row-0.5;
  object[index].y *= Real(x)/cuda_row-0.5;
})

cuFuncc(multiplyy,(complexFormat* object),(cuComplex* object),((cuComplex*)object),{
  cuda1Idx();
  int y = index%cuda_column;
  object[index].x *= Real(y)/cuda_row-0.5;
  object[index].y *= Real(y)/cuda_row-0.5;
})

cuFuncc(calcPartial,(complexFormat* object, complexFormat* Fn, Real* pattern, Real* beamstop),(cuComplex* object, cuComplex* Fn, Real* pattern, Real* beamstop),((cuComplex*)object,(cuComplex*)Fn,pattern,beamstop),{
  cuda1Idx();
  if(beamstop[index] > 0.5){
    object[index].x = 0;
    return;
  }
  Real ret;
  auto fntmp = Fn[index];
  Real fnmod2 = fntmp.x*fntmp.x + fntmp.y*fntmp.y;
  ret = fntmp.x*object[index].y - fntmp.y*object[index].x;
  Real fact = pattern[index]+DELTA;
  if(fact<0) fact = 0;
  ret*=1-sqrt(fact/(fnmod2+DELTA));
  object[index].x = ret;
})
