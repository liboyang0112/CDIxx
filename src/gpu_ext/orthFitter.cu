#include "cudaDefs.h"

__global__ void calcLambdas(int rows, double* lambdas, double step_lambda, double* matrix, double* bi, bool debug = 0){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= rows) return;
  double tmp = lambdas[y];
  for(int x = y; x < rows; x++){
    tmp -= step_lambda*matrix[x*(x+1)/2+y]*bi[x];
  }
  if(tmp < 0) tmp = 0;
  lambdas[y] = tmp;
}
__global__ void calcbs(int rows, double* bi, double* matrix, double* lambdas, double* prods){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= rows) return;
  double grad = 0;
  for(int y = 0; y <= x; y++){
    grad -= matrix[x*(x+1)/2+y]*lambdas[y];
  }
  bi[x] = prods[x]-0.5*grad;
}
void runIter_cu(int n, int niter, Real step_lambda, double* bi, double* prods, double* matrix){
  int sz = n*sizeof(double);
  int szmat = n*(n+1)/2*sizeof(double);
  double *lambdas = (double*)memMngr.borrowCache(sz); // lagrangian multiplier
  double *d_bi = (double*)memMngr.borrowCache(sz);
  double *d_prods = (double*)memMngr.borrowCache(sz);
  double *d_matrix = (double*)memMngr.borrowCache(szmat);
  cudaMemcpy(d_bi,bi,sz,cudaMemcpyHostToDevice);
  cudaMemcpy(d_prods,prods,sz,cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix,matrix,szmat,cudaMemcpyHostToDevice);
  gpuErrchk(cudaMemset(lambdas, 0, sz));
  dim3 nthd;
  nthd.x= min(256,n);
  dim3 nblk;
  nblk.x = ceil(Real(n)/nthd.x);
  for(int iter = 0; iter < niter; iter++){
    calcLambdas<<<nblk,nthd>>>(n, lambdas, step_lambda, d_matrix, d_bi, iter == niter - 1);
    calcbs<<<nblk,nthd>>>(n, d_bi, d_matrix, lambdas, d_prods);
  }
  cudaMemcpy(bi,d_bi,sz,cudaMemcpyDeviceToHost);
  memMngr.returnCache(lambdas);
  memMngr.returnCache(d_bi);
  memMngr.returnCache(d_prods);
  memMngr.returnCache(d_matrix);
}
