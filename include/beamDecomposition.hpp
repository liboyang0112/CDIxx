#include "format.hpp"
#include <cstddef>
complexFormat** zernikeDecomposition(complexFormat* img, int maxn, int radius, complexFormat* coefficient = NULL, complexFormat* projected = NULL);
complexFormat** laguerreDecomposition(complexFormat* img, int maxn, int maxl, int radius, complexFormat* coefficient = NULL, complexFormat* projected = NULL);
void zernike_reconstruct(void* handle_ptr, complexFormat* phi_out, Real radius) ;
void multiplyZernikeConj(complexFormat* store, complexFormat* data, Real pupilsize, int n, int m);
void multiplyLaguerreConj(complexFormat* store,complexFormat* data, Real pupilsize, int n, int m);
void addZernike(complexFormat* store, complexFormat coefficient, Real pupilsize, int n, int m);
void addLaguerre(complexFormat* store, complexFormat coefficient, Real pupilsize, int n, int m);
void multiplyHermit(complexFormat* store, complexFormat* data, Real pupilsize, int n, int m);
void* zernike_init(int width, int height, int maxN, int max_blocks);
void zernike_destroy(void* handle_ptr);
complexFormat* zernike_compute(void* handle_ptr, complexFormat* phi, Real cx, Real cy, Real radius);
complexFormat* zernike_coeff(void* handle_ptr);
void* laguerre_init(int width, int height, int maxN, int maxM, int max_blocks);
void laguerre_destroy(void* handle_ptr);
complexFormat* laguerre_compute(void* handle_ptr, complexFormat* phi, Real cx, Real cy, Real radius);
complexFormat* laguerre_coeff(void* handle_ptr);
void laguerre_reconstruct(void* handle_ptr, complexFormat* phi_out, Real radius);
