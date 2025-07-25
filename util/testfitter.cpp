#include "fmt/core.h"
#include "orthFitter.hpp"
#include "memManager.hpp"
#include <cstring>
#include <stdio.h>
const int N = 2;

Real innerProd(void* a, void* b, void* c){
  Real sum = 0;
  for(int i = 0; i < N; i++){
    if(c && ((bool*)c)[i]) sum+=((Real*)a)[i]*((Real*)b)[i];
  }
  return sum;
}

void mult(void* a, Real b){
  for(int i = 0; i < N; i++){
    ((Real*)a)[i] *= b;
  }
}

void add(void* a, void* b, Real c){
  for(int i = 0; i < N; i++){
    ((Real*)a)[i] += ((Real*)b)[i]*c;
  }
}

void* createCache(void* b){
  size_t sz = ccmemMngr.getSize(b);
  void* a = ccmemMngr.borrowCache(sz);
  memcpy(a, b, sz);
  return a;
}
void deleteCache(void* b){
  ccmemMngr.returnCache(b);
}

int main(){
  Real** vectors = (Real**)ccmemMngr.borrowCache(N*sizeof(Real*));
  for(int i = 0; i < N; i++){
    vectors[i] = (Real*) ccmemMngr.borrowCache(N*sizeof(Real));
  }
  vectors[0][0] = 1;
  vectors[0][1] = 0;
  vectors[1][0] = 1;
  vectors[1][1] = 2;
  Real right[N] = {0,59};
  double* out = (double*) ccmemMngr.borrowCache(N*sizeof(double));
  Fit(out, N, (void**)vectors, right, innerProd, mult, add, createCache, deleteCache);
  fmt::println("solved: {:f},{:f}", out[0], out[1]);
 }
