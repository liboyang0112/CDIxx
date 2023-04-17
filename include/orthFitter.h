#include "format.h"
void runIter_cu(int n, int niter, int niter1, Real step_lambda, Real step_bi, double* bi, double* prods, double* matrix);
void Fit(double* out, int n, void** vectors, void* right, Real (*innerProd)(void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void (deleteCache)(void*), bool renorm = 1);
