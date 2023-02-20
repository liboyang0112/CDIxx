#ifndef __COMMON_H__
#define __COMMON_H__
#include "format.h"

// Declare the variables

using namespace std;
static const int mergeDepth = 1; //use it only when input image is integers
static const Real scale = 1;

void *readComplexImage(const char* name);
void writeComplexImage(const char* name, void* data, int row, int column);
Real *readImage(const char* name, int &row, int &col, bool isFrequency = 0);
void getNormSpectrum(const char* fspectrum, const char* ccd_response, Real &startLambda, int &nlambda, Real *& outlambda, Real *& outspectrum);


#endif
