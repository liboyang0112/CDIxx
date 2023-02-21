#include "cudaDefs.h"

__global__ void calcLambdas(double* lambdas, double step_lambda, double* matrix, double* bi){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= cuda_row) return;
  double tmp = lambdas[y];
  for(int x = y; x < cuda_row; x++){
    tmp -= step_lambda*matrix[x*(x+1)/2+y]*bi[x];
  }
  if(tmp < 0) tmp = 0;
  lambdas[y] = tmp;
}
__global__ void calcGrads(double* grads, double* matrix, double* lambdas){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= cuda_row) return;
  grads[x] = 0;
  for(int y = 0; y <= x; y++){
    grads[x] -= matrix[x*(x+1)/2+y]*lambdas[y];
  }
}
__global__ void calcbi(double* bi, double* grads, double* prods, double step_bi){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= cuda_row) return;
  bi[x] -= step_bi*(2*(bi[x]-prods[x])+grads[x]);
}
void runIter_cu(int n, int niter, int niter1, Real step_lambda, Real step_bi, double* bi, double* prods, double* matrix){
  int sz = n*sizeof(double);
  int szmat = n*(n+1)/2*sizeof(double);
  double *lambdas = (double*)memMngr.borrowCache(sz); // lagrangian multiplier
  double *d_bi = (double*)memMngr.borrowCache(sz);
  double *d_prods = (double*)memMngr.borrowCache(sz);
  double *d_matrix = (double*)memMngr.borrowCache(szmat);
  double *grads = (double*)memMngr.borrowCache(sz);
  cudaMemcpy(d_bi,bi,sz,cudaMemcpyHostToDevice);
  cudaMemcpy(d_prods,prods,sz,cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix,matrix,szmat,cudaMemcpyHostToDevice);
  gpuErrchk(cudaMemset(lambdas, 0, sz));
  init_cuda_image(n,1);
  dim3 nthd;
  nthd.x= min(256,n);
  dim3 nblk;
  nblk.x = ceil(Real(n)/nthd.x);
  for(int iter = 0; iter < niter; iter++){
    calcLambdas<<<nblk,nthd>>>(lambdas, step_lambda, d_matrix, d_bi);
    calcGrads<<<nblk,nthd>>>(grads, d_matrix, lambdas);
    for(int iter1 = 0; iter1 < niter1; iter1 ++){
      calcbi<<<nblk,nthd>>>(d_bi, grads, d_prods, step_bi);
    }
  }
  cudaMemcpy(bi,d_bi,sz,cudaMemcpyDeviceToHost);
  memMngr.returnCache(lambdas);
  memMngr.returnCache(grads);
  memMngr.returnCache(d_bi);
  memMngr.returnCache(d_prods);
  memMngr.returnCache(d_matrix);
}
