#include "string.h"
#include "orthFitter.h"
#include "memManager.h"

void runIter(int n, int niter, Real step_lambda, double* bi, double* prods, double* matrix){
  double *lambdas = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // lagrangian multiplier
  memset(lambdas, 0, n*sizeof(double));
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
  double *matrix = (double*)ccmemMngr.borrowCache(n*(n+1)/2*sizeof(double));  // b_i = M_ij*a_j
  void **orthedVector = (void**)ccmemMngr.borrowCache(n*sizeof(void*));
  memset(matrix, 0, n*(n+1)/2*sizeof(double));
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

