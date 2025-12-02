#include "format.hpp"
#include <stdint.h>
void applyGaussConv(Real* input, Real* output, Real* gaussMem, Real sigma, int size = 0);
complexFormat findMiddle(complexFormat* d_in, int num = 0);
complexFormat findMiddle(Real* d_in, int num = 0);
void shiftMiddle(complexFormat* wave);
void shiftWave(complexFormat* wave, Real shiftx, Real shifty);
void shiftWave(int plan, complexFormat* wave, Real shiftx, Real shifty);
void readComplexWaveFront(const char* intensityFile, const char* phaseFile, Real* &d_intensity, Real* &d_phase, int &objrow, int &objcol);
uint32_t* createMaskMap(Real* refMask, int &pixCount, int row, int col, int mrow, int mcol, int shiftx, int shifty);
void createCircleMask(Real* data, Real x0, Real y0, Real r, bool isFreq = 0);
void convolute(complexFormat* store, complexFormat* input1, complexFormat* input2, complexFormat* cache, int upsample, int handle);
