#include "tvFilter.h"
#include "memManager.h"
#include "cudaConfig.h"
#include "cuPlotter.h"
#define swap(x, y) \
  auto x##swapTMPVariable = x;\
  x = y;\
  y = x##swapTMPVariable

cuFunc(partialx, (Real* b, Real* p), (b,p),{
  cudaIdx()
  if(x == cuda_row-1) p[index] = b[index]-b[index%cuda_column];
  else p[index] = b[index]-b[index+cuda_column];
})
cuFunc(partialy, (Real* b, Real* p), (b,p),{
  cudaIdx()
  if(y == cuda_column-1) p[index] = b[index]-b[index-cuda_column+1];
  else p[index] = b[index]-b[index+1];
})
cuFunc(diffMax, (Real* p, Real* q), (p,q),{
  cudaIdx()
  Real mod = hypot(p[index],q[index]);
  if(mod <= 1) return;
  p[index] /= mod;
  q[index] /= mod;
})
cuFunc(calcLpq, (Real* out, Real* p, Real* q), (out,p,q),{
  cudaIdx()
  Real tmp = p[index]+q[index];
  if(x >= 1) tmp -= p[index-cuda_column];
  else tmp-=p[index+(cuda_row-1)*cuda_column];
  if(y >= 1) tmp -= q[index-1];
  else tmp-=q[index+cuda_column-1];
  out[index] = tmp;
})
void FISTA(Real* b, Real* output, Real lambda, int niter, void (applyC)(Real*, Real*)){
  size_t sz = memMngr.getSize(b);
  Real tk = 0.5+sqrt(1.25);
  Real* pij = (Real*)memMngr.borrowCache(sz);
  Real* qij = (Real*)memMngr.borrowCache(sz);
  Real* lpq = (Real*)memMngr.borrowCache(sz);
  Real* pijprev = (Real*)memMngr.borrowCache(sz);
  Real* qijprev = (Real*)memMngr.borrowCache(sz);
  cudaMemset(pij, 0, sz);
  cudaMemset(qij, 0, sz);
  bool replaceout = 0;
  if(output == b) {
    replaceout = 1;
    output = (Real*)memMngr.borrowCache(sz);
  }
  if(applyC) applyC(b, output);
  else cudaMemcpy(output, b, sz, cudaMemcpyDeviceToDevice);
  for(int iter = 0; iter < niter ; iter++){
    swap(pij, pijprev);
    swap(qij, qijprev);
    cudaF(applyNorm, output, 0.125/lambda);
    cudaF(partialx, output, pij);
    cudaF(partialy, output, qij);
    cudaF(add, pij, pijprev, 1);
    cudaF(add, qij, qijprev, 1);
    cudaF(diffMax, pij, qij);
    Real tkp1 = 0.5+sqrt(0.25+tk*tk);
    Real fact1 = (tk-1)/tkp1;
    tk = tkp1;
    cudaF(applyNorm, pij, 1+fact1);
    cudaF(applyNorm, qij, 1+fact1);
    cudaF(add, pij, pijprev, -fact1);
    cudaF(add, qij, qijprev, -fact1);
    cudaF(calcLpq, lpq, pij, qij);
    cudaF(add, output, b, lpq, -lambda);
    if(applyC) applyC(output, output);
  }
  memMngr.returnCache(pij);
  memMngr.returnCache(qij);
  memMngr.returnCache(lpq);
  memMngr.returnCache(pijprev);
  memMngr.returnCache(qijprev);
  if(replaceout){
    cudaMemcpy(b, output, sz, cudaMemcpyDeviceToDevice);
    memMngr.returnCache(output);
  }
};
