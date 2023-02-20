#include "format.h"
Real* Fit(int n, void** vectors, void* right, Real (*innerProd)(void*, void*), void (*mult)(void*, Real), void (*add)(void*, void*, Real), void* (createCache)(void*), void (deleteCache)(void*), bool renorm = 1);
