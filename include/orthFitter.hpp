#include "format.hpp"
//#define useLapack
void runIter_cu(int n, int niter, Real step_lambda, double* bi, double* prods, double* matrix);
void Fit(double* out, int n, void** vectors, void* right, Real (*innerProd)(void*, void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void (deleteCache)(void*), bool renorm = 1, void* param = 0);
void Fit_fast(double* out, int n, void** vectors, void* right, Real (*innerProd)(void*, void*, void*), void* param);
void Fit_fast_matrix(double* out, int n, double* matrix, double* bi);
void calcbs(double* bi, double* matrix, double* lambdas, double* prods);
void calcLambdas (double* lambdas, double step_lambda, double* matrix, double* bi);
void fillMatrix(double* matrix, double* matrix_ext);
void calcas_fast(double* bi, double* matrix, double* lambdas, double* out);
void calcLambdas_fast(double* lambdas, double step_lambda, double* out);
