#include "format.h"
//#define useLapack
void runIter_cu(int n, int niter, Real step_lambda, double* bi, double* prods, double* matrix);
void Fit(double* out, int n, void** vectors, void* right, Real (*innerProd)(void*, void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void (deleteCache)(void*), bool renorm = 1, void* param = 0);
void Fit_fast(double* out, int n, void** vectors, void* right, Real (*innerProd)(void*, void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void deleteCache(void*), bool renorm, void* param);
void runIter_fast_cu(int n, int niter, double step_lambda, double* out, double* matrix);
