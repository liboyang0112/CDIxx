#include "cudaDefs_h.cu"

cuFunc(calcLambdas, (double* lambdas, double step_lambda, double* matrix, double* bi), (lambdas, step_lambda, matrix, bi), {
  cuda1Idx();
  double tmp = lambdas[index];
  for(int x = index; x < cuda_row; x++){
    tmp -= step_lambda*matrix[x*(x+1)/2+index]*bi[x];
  }
  if(tmp < 0) tmp = 0;
  lambdas[index] = tmp;
})
cuFunc(calcbs, (double* bi, double* matrix, double* lambdas, double* prods), (bi, matrix, lambdas, prods), {
  cuda1Idx();
  double grad = 0;
  for(int y = 0; y <= index; y++){
    grad -= matrix[index*(index+1)/2+y]*lambdas[y];
  }
  bi[index] = prods[index]-0.5*grad;
})
__forceinline__ __device__ int d_getRFPidx(int i, int j, int n){
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
cuFunc(calcLambdas_fast,(double* lambdas, double step_lambda, double* out), (lambdas, step_lambda, out), {
  cuda1Idx()
  double tmp = lambdas[index]+step_lambda*out[index];
  if(tmp > 0) tmp = 0;
  lambdas[index] = tmp;
})
cuFunc(calcas_fast,(double* bi, double* matrix, double* lambdas, double* out), (bi, matrix, lambdas, out), {
  cuda1Idx()
  double tmp = bi[index];
  for(int j = 0; j < cuda_row; j++){
    tmp -= matrix[j*cuda_row+index]*lambdas[j];
  }
  out[index] = tmp;
})
cuFunc(fillMatrix,(double* matrix, double* matrix_ext), (matrix, matrix_ext), {
  cudaIdx();
  matrix_ext[index] = matrix[d_getRFPidx(y,x,cuda_row)];
})
