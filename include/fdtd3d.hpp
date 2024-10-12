#define n_PML 1
#define k_PML 0.0035
#define secondOrder
#ifdef secondOrder
#define fact1 1./24
#define fact2 9./8
#endif
#include "format.hpp"
#include <stddef.h>
const Real b_PML = 1-k_PML*n_PML;
void updateH(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez);
void updateE(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez);
void applyPMLx0_d (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez);
void applyPMLx1_d (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, int nx);
void applyPMLx1 (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* ExBdx1, Real* EyBdx1, Real* EzBdx1, int nx);
void applyPMLx1post (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* ExBdx1, Real* EyBdx1, Real* EzBdx1, int nx);
void applyPMLx0 (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HyBdx0, Real* HzBdx0, int nx);
void applyPMLy1 (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* ExBdy1, Real* EzBdy1, int ny);
void applyPMLy0 (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HxBdy0, Real* HzBdy0, int ny);
void applyPMLz1 (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* ExBdz1, Real* EyBdz1, int nz);
void applyPMLz0(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HxBdz0, Real* HyBdz0, int nz);
void applySource(Real* Ez, size_t idx, Real val);
