#include "string.h"
#include "orthFitter.h"
#include "memManager.h"

void runIter(int n, int niter, int niter1, Real step_lambda, Real step_bi, double* bi, double* prods, double* matrix){
  double *lambdas = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // lagrangian multiplier
  double *grads = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // lagrangian multiplier
  memset(lambdas, 0, n*sizeof(double));
  for(int iter = 0; iter < niter; iter++){
    for(int j = 0; j < n; j++){
      for(int i = j; i < n; i++){
        lambdas[j] -= step_lambda*matrix[i*(i+1)/2+j]*bi[i];
      }
      if(lambdas[j]<0 && j!=0) lambdas[j] = 0;  // constrains: a_i > 0;
    }
    for(int i = 0; i < n; i++){
      grads[i] = 0;
      for(int j = 0; j <= i; j++){
        grads[i] -= matrix[i*(i+1)/2+j]*lambdas[j];
      }
    }
    for(int iter1 = 0; iter1 < niter1; iter1 ++){
      for(int i = 0; i < n; i++){
        bi[i] -= step_bi*(2*(bi[i]-prods[i])+grads[i]);
      }
    }
  }
  ccmemMngr.returnCache(lambdas);
  ccmemMngr.returnCache(grads);
}

void Fit(double* out, int n, void** vectors, void* right, Real (*innerProd)(void*, void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void deleteCache(void*), bool renorm, void* param){
  double *bi = (double*)ccmemMngr.borrowCache(n*sizeof(double)); //orthogalized ai
  double *ni = (double*)ccmemMngr.borrowCache(n*sizeof(double));  //normalization of each vector
  double *matrix = (double*)ccmemMngr.borrowCache(n*(n+1)/2*sizeof(double));  // b_i = M_ij*a_j
  void **orthedVector = (void**)ccmemMngr.borrowCache(n*sizeof(void*));
  memset(matrix, 0, n*(n+1)/2*sizeof(double));
  for(int i = 0; i < n; i++){
    if(renorm) ni[i] = sqrt(innerProd(vectors[i],vectors[i],param));
    mult(vectors[i], 1./ni[i]);
    orthedVector[i] = createCache(vectors[i]);
  }

  Real maxnorm = 1;

  matrix[0] = 1.;
  //calculate matrix and orthedVector

  for(int i = 1; i < n; i++){
    for(int j = 0; j < i; j++){
      Real prod = innerProd(vectors[i], orthedVector[j],param);
      add(orthedVector[i],orthedVector[j], -prod);
      for(int k = 0; k <= j; k++){
        matrix[i*(i+1)/2+k]-=prod*matrix[j*(j+1)/2+k];
      }
      //norm-=pow(prod,2);
    }
    //if(norm<0) {
    //  printf("norm is negative, this is impossible, please check!\n");
    //  abort();
    //}
    double norms = sqrt(innerProd(orthedVector[i],orthedVector[i],param));
    int idxii = i*(i+3)/2;
    matrix[idxii] = 1./norms;
    if(maxnorm < matrix[idxii]) maxnorm = matrix[idxii];
    mult(orthedVector[i], 1./norms);
    for(int j = 0; j < i; j++){
      matrix[idxii-i+j] /= norms;
    }
  }

  Real step_lambda = 2./(n*maxnorm*maxnorm);
  //printf("step_lambda = %e\n", step_lambda);
  Real step_bi = 1.5/n;
  int niter = 10000;
  int niter1 = 5;
  //now lets solve dual problem
  double *prods = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // b_i without positive constraints
  for(int i = 0; i < n; i++){
    bi[i] = prods[i] = innerProd(right,orthedVector[i],param);
  }
  runIter_cu(n, niter, niter1, step_lambda, step_bi, bi, prods, matrix);
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
      mult(vectors[i], ni[i]);
      out[i] /= ni[i];
    }
    deleteCache(orthedVector[i]);
  }
  ccmemMngr.returnCache(ni);
  ccmemMngr.returnCache(orthedVector);
}

