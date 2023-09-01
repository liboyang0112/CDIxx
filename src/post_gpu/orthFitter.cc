#include "string.h"
#include "orthFitter.h"
#include "memManager.h"
#ifdef useLapack
#include "lapacke.h"
#else
#include "matrixInverse.h"
#endif

void runIter(int n, int niter, Real step_lambda, double* bi, double* prods, double* matrix){
  double *lambdas = (double*)ccmemMngr.borrowCleanCache(n*sizeof(double)); // lagrangian multiplier
  for(int iter = 0; iter < niter; iter++){
    for(int j = 0; j < n; j++){
      for(int i = j; i < n; i++){
        lambdas[j] -= step_lambda*matrix[i*(i+1)/2+j]*bi[i];
      }
      if(lambdas[j]<0 && j!=0) lambdas[j] = 0;  // constrains: a_i > 0;
    }
    for(int i = 0; i < n; i++){
      bi[i] = prods[i];
      for(int j = 0; j <= i; j++){
        bi[i] += 0.5*matrix[i*(i+1)/2+j]*lambdas[j];
      }
    }
  }
  ccmemMngr.returnCache(lambdas);
}

void Fit(double* out, int n, void** vectors, void* right, Real (*innerProd)(void*, void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void deleteCache(void*), bool renorm, void* param){
  double *bi = (double*)ccmemMngr.borrowCache(n*sizeof(double)); //orthogalized ai
  double *ni = (double*)ccmemMngr.borrowCache(n*sizeof(double));  //normalization of each vector
  double *matrix = (double*)ccmemMngr.borrowCleanCache(n*(n+1)/2*sizeof(double));  // b_i = M_ij*a_j
  void **orthedVector = (void**)ccmemMngr.borrowCache(n*sizeof(void*));
  for(int i = 0; i < n; i++){
    orthedVector[i] = createCache(vectors[i]);
    if(renorm) {
      ni[i] = sqrt(innerProd(orthedVector[i],orthedVector[i],param));
      mult(orthedVector[i], 1./ni[i]);
    }
  }

  double maxnorm = 1;

  //eye matrix;
  for(int i = 0; i < n; i++){
    matrix[i*(i+3)/2] = 1;
  }
  //calculate matrix and orthedVector
  for(int iter = 0; iter < 1; iter++){
    for(int i = 1; i < n; i++){
      for(int j = 0; j < i; j++){
        Real prod = innerProd(orthedVector[i], orthedVector[j],param);
        add(orthedVector[i],orthedVector[j], -prod);
        for(int k = 0; k <= j; k++){
          matrix[i*(i+1)/2+k]-=prod*matrix[j*(j+1)/2+k];
        }
        //norm-=sq(prod);
      }
      //if(norm<0) {
      //  printf("norm is negative, this is impossible, please check!\n");
      //  abort();
      //}
      double norms = sqrt(innerProd(orthedVector[i],orthedVector[i],param));
      int idxii = i*(i+3)/2;
      mult(orthedVector[i], 1./norms);
      for(int j = 0; j <= i; j++){
        matrix[idxii-i+j] /= norms;
        if(maxnorm < matrix[idxii-i+j]) maxnorm = matrix[idxii-i+j];
      }
    }
    for(int iv = 0; iv < n; iv++){
      deleteCache(orthedVector[iv]);
      orthedVector[iv] = createCache(vectors[0]);
      mult(orthedVector[iv], matrix[iv*(iv+1)/2]/ni[0]);
      for(int i = 1; i <= iv; i++){
        add(orthedVector[iv], vectors[i], matrix[iv*(iv+1)/2+i]/ni[i]);
        //printf("%f, ", matrix[iv*(iv+1)/2+i]);
      }
      //printf("\n");
    }
    printf("inner prod= %f\n", innerProd(vectors[0], orthedVector[n-1],param)/ni[0]);
  }

  Real step_lambda = 30./(maxnorm*maxnorm*n);
  printf("step_lambda = %e\n", step_lambda);
  int niter = 30000;
  //now lets solve dual problem
  double *prods = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // b_i without positive constraints
  for(int i = 0; i < n; i++){
    bi[i] = prods[i] = innerProd(right,orthedVector[i],param);
  }
  runIter_cu(n, niter, step_lambda, bi, prods, matrix);
  ccmemMngr.returnCache(prods);
  //calculate a.
  memset(out, 0, n*sizeof(double));
  for(int j = 0; j < n; j++){
    for(int i = j; i < n; i++){
      int rs = i*(i+1)/2;
      out[j] += matrix[rs+j]*bi[i];
    }
  }
  ccmemMngr.returnCache(matrix);
  ccmemMngr.returnCache(bi);
  for(int i = 0; i < n; i++){
    if(renorm) {
      out[i] /= ni[i];
    }
    deleteCache(orthedVector[i]);
  }
  ccmemMngr.returnCache(ni);
  ccmemMngr.returnCache(orthedVector);
}

int getRFPidx(int i, int j, int n){
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

void runIter_fast(int n, int niter, double step_lambda, double* bi, double* out, double* matrix){
  double *lambda = (double*)ccmemMngr.borrowCleanCache(n*sizeof(double)); // lambdas
  memcpy(bi, out, n*sizeof(double));
  for(int iter = 0; iter < niter; iter++){
    //update lambda
    for(int i = 0; i < n; i++){
      lambda[i] += step_lambda*out[i];
      if(lambda[i] > 0) lambda[i] = 0;
    }
    //update x
    for(int i = 0; i < n; i++){
      out[i] = bi[i];
      for(int j = 0; j < n; j++){
        out[i] -= matrix[getRFPidx(i,j,n)]*lambda[j];
      }
    }
  }
}

void Fit_fast(double* out, int n, void** vectors, void* right, Real (*innerProd)(void*, void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void deleteCache(void*), bool renorm, void* param){
  double *bi = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // ATb
  double *ni = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // norms
  double *matrix = (double*)ccmemMngr.borrowCleanCache(
#ifdef useLapack
      n*(n+1)/2
#else
      n*n
#endif
      *sizeof(double));  // ATA
  if(renorm) {
    for(int i = 0; i < n; i++){
      ni[i] = sqrt(innerProd(vectors[i],vectors[i],param));
      mult(vectors[i], 1./ni[i]);
    }
  }
  for(int i = 0; i < n; i++){
    for(int j = 0; j <= i; j++){
#ifdef useLapack
      int idx = getRFPidx(i,j,n);
      matrix[getRFPidx(i,j,n)]
#else
      matrix[j*n+i] = matrix[i*n+j]
#endif
      =innerProd(vectors[i], vectors[j],param);
    }
    bi[i] = innerProd(vectors[i],right, param);
  }
#ifdef useLapack
  int errIdx;
  printf("inverting matrix!\n");
  if(errIdx = LAPACKE_dpftrf(LAPACK_ROW_MAJOR, 'N', 'L', n, matrix)){
    for(int i = 0; i < n; i++){
      printf("%f, ",matrix[getRFPidx(i,i,n)]);
    }
    printf("\n");
    fprintf(stderr, "Error: lapack factorization failed for idx %d, %f!\n", errIdx, matrix[getRFPidx(errIdx+1,errIdx+1,n)]);
    abort();
  }   
  if(LAPACKE_dpftri(LAPACK_ROW_MAJOR, 'N', 'L', n, matrix)){
    fprintf(stderr, "Error: lapack inverse failed!\n");
    abort();
  }   
#else
  inverseMatrixEigen(matrix, n);
#endif
  double maxval = 0;
  for(int i = 0; i < n; i++){
    double val = 0;
    for(int j = 0; j < n; j++){
      double &val1 = matrix[
#ifdef useLapack
        getRFPidx(i,j,n)
#else
        i+j*n
#endif
      ];
      out[i] += val1*bi[j];
      val+=fabs(val1);
    }
    if(maxval < val) maxval = val;
  }
  double step_lambda = 1./maxval;
  int niter = 60000;
  //runIter_fast(n, niter, step_lambda, bi, out, matrix);
  runIter_fast_cu(n, niter, step_lambda, out, matrix);
  if(renorm) {
    for(int i = 0; i < n; i++){
      mult(vectors[i], ni[i]);
      out[i] /= ni[i];
    }
  }
  ccmemMngr.returnCache(matrix);
  ccmemMngr.returnCache(bi);
}
