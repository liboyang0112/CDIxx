#include "string.h"
#include "orthFitter.h"
#include "memManager.h"

Real* Fit(int n, void** vectors, void* right, Real (*innerProd)(void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void deleteCache(void*), bool renorm){
  Real *ai = (Real*)ccmemMngr.borrowCache(n*sizeof(Real));  //output
  double *bi = (double*)ccmemMngr.borrowCache(n*sizeof(double)); //orthogalized ai
  double *ni = (double*)ccmemMngr.borrowCache(n*sizeof(double));  //normalization of each vector
  double *prods = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // b_i without positive constraints
  double *grads = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // gradiants
  double *lambdas = (double*)ccmemMngr.borrowCache(n*sizeof(double)); // lagrangian multiplier
  double **matrix = (double**)ccmemMngr.borrowCache(n*sizeof(double*));  // b_i = M_ij*a_j
  void **orthedVector = (void**)ccmemMngr.borrowCache(n*sizeof(void*));
  memset(lambdas, 0, n*sizeof(double));
  memset(ai, 0, n*sizeof(Real));
  for(int i = 0; i < n; i++){
    matrix[i] = (double*)ccmemMngr.borrowCache((i+1)*sizeof(double));
    memset(matrix[i], 0, (i+1)*sizeof(double));
    if(renorm) ni[i] = sqrt(innerProd(vectors[i],vectors[i]));
    mult(vectors[i], 1./ni[i]);
    orthedVector[i] = createCache(vectors[i]);
  }

  Real maxnorm = 0;

  matrix[0][0] = 1.;
  //calculate matrix and orthedVector

  for(int i = 1; i < n; i++){
    for(int j = 0; j < i; j++){
      Real prod = innerProd(vectors[i], orthedVector[j]);
      add(orthedVector[i],orthedVector[j], -prod);
      for(int k = 0; k <= j; k++){
        matrix[i][k]-=prod*matrix[j][k];
      }
      //norm-=pow(prod,2);
    }
    //if(norm<0) {
    //  printf("norm is negative, this is impossible, please check!\n");
    //  abort();
    //}
    double norms = sqrt(innerProd(orthedVector[i],orthedVector[i]));
    matrix[i][i] = 1./norms;
    if(maxnorm < matrix[i][i]) maxnorm = matrix[i][i];
    mult(orthedVector[i], 1./norms);
    for(int j = 0; j < i; j++){
      matrix[i][j] /= norms;
    }
  }

  Real step_lambda = 1./(n*maxnorm*maxnorm);
  printf("step_lambda = %e\n", step_lambda);
  Real step_bi = 0.5/n;
  int niter = 10000;
  int niter1 = 100;
  //now lets solve dual problem
  for(int i = 0; i < n; i++){
    bi[i] = prods[i] = innerProd(right,orthedVector[i]);
  }
  for(int iter = 0; iter < niter; iter++){
    for(int i = 0; i < n; i++){
      for(int j = 0; j < i+1; j++){
        lambdas[j] -= step_lambda*matrix[i][j]*bi[i];
      }
    }
    if(iter == niter-1) printf("lambda = %f,%f\n",lambdas[0],lambdas[n-1]);
    for(int i = 1; i < n; i++){
      if(lambdas[i]<0) lambdas[i] = 0;  // constrains: a_i > 0;
    }
    for(int iter1 = 0; iter1 < niter1; iter1 ++){
      double gradmax = 0;
      for(int i = 0; i < n; i++){
        grads[i] = 2*bi[i]-2*prods[i];
      }
      for(int i = 0; i < n; i++){
        for(int j = 0; j < i+1; j++){
          grads[i] -= matrix[i][j]*lambdas[j];
        }
        if(fabs(grads[i]) > gradmax) gradmax = fabs(grads[i]);
      }
      //if(iter1 > niter1-5 && iter == niter-1) printf("bi=%f,%f\n",bi[0],bi[1]);
      if(gradmax < 1e-15) break;
      for(int i = 0; i < n; i++){
         bi[i] -= step_bi*grads[i];
      }
    }
  }
  //calculate a.
  for(int i = 0; i < n; i++){
    for(int j = 0; j < i+1; j++){
      ai[j] += matrix[i][j]*bi[i];
    }
  }
  for(int i = 0; i < n; i++){
    if(renorm) {
      mult(vectors[i], ni[i]);
      ai[i] /= ni[i];
    }
    ccmemMngr.returnCache(matrix[i]);
    deleteCache(orthedVector[i]);
  }
  ccmemMngr.returnCache(bi);
  ccmemMngr.returnCache(ni);
  ccmemMngr.returnCache(grads);
  ccmemMngr.returnCache(prods);
  ccmemMngr.returnCache(matrix);
  ccmemMngr.returnCache(orthedVector);
  return ai;
}
