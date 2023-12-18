#include "cudaDefs.hpp"
#include "cudaConfig.hpp"
#include "orthFitter.hpp"

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
  double *lambdas = (double*)memMngr.borrowCleanCache(sz); // lagrangian multiplier
  double *d_bi = (double*)memMngr.borrowCache(sz);
  double *d_prods = (double*)memMngr.borrowCache(sz);
  int szmat = n*(n+1)/2*sizeof(double);
  double *d_matrix = (double*)memMngr.borrowCache(szmat);
  cudaMemcpy(d_bi,bi,sz,cudaMemcpyHostToDevice);
  cudaMemcpy(d_prods,prods,sz,cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix,matrix,szmat,cudaMemcpyHostToDevice);
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
__device__ int d_getRFPidx(int i, int j, int n){
  if(i < j){
    int tmp = i;
    i = j;
    j = tmp;
  }
  if(n%2==1){
    int half = n/2+1;
    if(j < half){
      return i*half+j;
    }else{
      return (j-half)*half+i-half+1;
    }
  }else{
    int half = n/2;
    if(j < half){
      return (i+1)*half+j;
    }else{
      return (j-half)*half+i-half;
    }
  }
}
__global__ void calcLambdas(int rows, double* lambdas, double step_lambda, double* out){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= rows) return;
  double tmp = lambdas[y]+step_lambda*out[y];
  if(tmp > 0) tmp = 0;
  lambdas[y] = tmp;
}
__global__ void calcas(int rows, double* bi, double* matrix, double* lambdas, double* out){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= rows) return;
  double tmp = bi[x];
  for(int j = 0; j < rows; j++){
    tmp -= matrix[j*rows+x]*lambdas[j];
  }
  out[x] = tmp;
}
__global__ void fillMatrix(int rows, double* matrix, double* matrix_ext){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= rows*rows) return;
  matrix_ext[x] = matrix[d_getRFPidx(x%rows,x/rows,rows)];
}
void runIter_fast_cu(int n, int niter, double step_lambda, double* out, double* matrix){
  size_t sz = n*sizeof(double);
  double *lambda = (double*)memMngr.borrowCleanCache(sz); // lambdas
  int szmat = 
#ifdef useLapack
    n*(n+1)/2
#else
    n*n
#endif
    *sizeof(double);
  double *d_out = (double*)memMngr.borrowCache(sz);
  double *d_bi = (double*)memMngr.borrowCache(sz);
  double *d_matrix = (double*)memMngr.borrowCache(szmat);
  cudaMemcpy(d_bi, out, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix, matrix, szmat, cudaMemcpyHostToDevice);
  dim3 nthd;
  nthd.x= min(256,n);
  dim3 nblk;
  nblk.x = ceil(Real(n)/nthd.x);
#ifdef useLapack
  double *d_matrix_ext = (double*)memMngr.borrowCache(n*n*sizeof(double));
  fillMatrix<<<ceil(Real(n*n)/nthd.x),256>>>(n, d_matrix, d_matrix_ext);
  memMngr.returnCache(d_matrix);
  d_matrix = d_matrix_ext;
#endif
  for(int iter = 0; iter < niter; iter++){
    calcLambdas<<<nblk,nthd>>>(n, lambda, step_lambda, d_out);
    calcas<<<nblk,nthd>>>(n, d_bi, d_matrix, lambda, d_out);
  }
  cudaMemcpy(out,d_out,sz,cudaMemcpyDeviceToHost);
  memMngr.returnCache(d_matrix);
  memMngr.returnCache(d_bi);
  memMngr.returnCache(d_out);
  memMngr.returnCache(lambda);
}
