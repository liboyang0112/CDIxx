#include "format.hpp"
void FISTA(Real* b, Real* output, Real lambda, int niter, void (applyC)(Real*,Real*));
void FISTA(complexFormat* b, complexFormat* output, Real lambda, int niter, void (applyC)(complexFormat*,complexFormat*));
void partialx (Real* b, Real* p);
void partialy (Real* b, Real* p);
void diffMax (Real* p, Real* q);
void calcLpq (Real* out, Real* p, Real* q);
void partialx (complexFormat* b, complexFormat* p);
void partialy (complexFormat* b, complexFormat* p);
void diffMax (complexFormat* p, complexFormat* q);
void calcLpq (complexFormat* out, complexFormat* p, complexFormat* q);
