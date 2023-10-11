#ifndef __COMMON_H__
#define __COMMON_H__
#include "format.h"

// Declare the variables

using namespace std;
static const int mergeDepth = 1; //use it only when input image is integers
static const Real scale = 1;

void writeComplexImage(const char* name, void* data, int row, int column);
Real *readImage(const char* name, int &row, int &col);
void getNormSpectrum(const char* fspectrum, const char* ccd_response, Real &startLambda, Real &endLambda, int &nlambda, double *& outlambda, double *& outspectrum);
void getRealSpectrum(const char* ccd_response, int nlambda, double* lambdas, double* spectrum);


#endif
