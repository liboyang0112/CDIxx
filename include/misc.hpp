#include "format.hpp"
void applyGaussConv(Real* input, Real* output, Real* gaussMem, Real sigma, int size = 0);
complexFormat findMiddle(complexFormat* d_in, int num = 0);
complexFormat findMiddle(Real* d_in, int num = 0);
void shiftMiddle(complexFormat* wave);
void shiftWave(complexFormat* wave, Real shiftx, Real shifty);
void readComplexWaveFront(const char* intensityFile, const char* phaseFile, Real* &d_intensity, Real* &d_phase, int &objrow, int &objcol);
