#include "format.h"
#include <cufft.h>
Real findSum(Real* d_in, int num_items, bool debug=false);
Real findMax(Real* d_in, int num_items);
Real findSumReal(complexFormat* d_in, int num_items);
Real findMod2Max(complexFormat* d_in, int num_items);
void initCub();
