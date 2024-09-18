#include "format.hpp"
void FISTA(Real* b, Real* output, Real lambda, int niter, void (applyC)(Real*,Real*));
void partialx (Real* b, Real* p);
void partialy (Real* b, Real* p);
void diffMax (Real* p, Real* q);
void calcLpq (Real* out, Real* p, Real* q);
