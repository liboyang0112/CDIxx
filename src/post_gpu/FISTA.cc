#include "tvFilter.h"
#include "cudaConfig.h"
#include <math.h>
#define swap(x, y) \
  auto x##swapTMPVariable = x;\
  x = y;\
  y = x##swapTMPVariable

void FISTA(Real* b, Real* output, Real lambda, int niter, void (applyC)(Real*, Real*)){
  size_t sz = memMngr.getSize(b);
  Real tk = 0.5+sqrt(1.25);
  Real* pij = (Real*)memMngr.borrowCleanCache(sz);
  Real* qij = (Real*)memMngr.borrowCleanCache(sz);
  Real* lpq = (Real*)memMngr.borrowCache(sz);
  Real* pijprev = (Real*)memMngr.borrowCache(sz);
  Real* qijprev = (Real*)memMngr.borrowCache(sz);
  bool replaceout = 0;
  if(output == b) {
    replaceout = 1;
    output = (Real*)memMngr.borrowCache(sz);
  }
  if(applyC) applyC(b, output);
  else myMemcpyD2D(output, b, sz);
  for(int iter = 0; iter < niter ; iter++){
    swap(pij, pijprev);
    swap(qij, qijprev);
    applyNorm( output, 0.125/lambda);
    partialx( output, pij);
    partialy( output, qij);
    add( pij, pijprev, 1);
    add( qij, qijprev, 1);
    diffMax( pij, qij);
    Real tkp1 = 0.5+sqrt(0.25+tk*tk);
    Real fact1 = (tk-1)/tkp1;
    tk = tkp1;
    applyNorm( pij, 1+fact1);
    applyNorm( qij, 1+fact1);
    add( pij, pijprev, -fact1);
    add( qij, qijprev, -fact1);
    calcLpq( lpq, pij, qij);
    add( output, b, lpq, -lambda);
    if(applyC) applyC(output, output);
  }
  memMngr.returnCache(pij);
  memMngr.returnCache(qij);
  memMngr.returnCache(lpq);
  memMngr.returnCache(pijprev);
  memMngr.returnCache(qijprev);
  if(replaceout){
    myMemcpyD2D(b, output, sz);
    memMngr.returnCache(output);
  }
};
