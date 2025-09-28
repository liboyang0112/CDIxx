#include "format.hpp"
#include <cstddef>
Real findSum(Real* d_in, int num = 0, void* out = NULL);
Real findRootSumSq(Real* d_in, int num = 0, void* out = NULL);
complexFormat findSum(complexFormat* d_in, int num = 0, void* out = NULL);
Real findMax(Real* d_in, int num = 0, void* out = NULL);
int findMax(int* d_in, int num = 0, void* out = NULL);
int findMin(int* d_in, int num = 0, void* out = NULL);
Real findSumReal(complexFormat* d_in, int num = 0, void* out = NULL);
Real findMod2Max(complexFormat* d_in, int num = 0, void* out = NULL);
void initCub();
