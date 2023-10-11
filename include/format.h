#ifndef __FORMAT_H__
#define __FORMAT_H__
#define Bits 16
#include <complex>

#define float_cv_format CV_32FC
#define VTK_TYPE VTK_FLOAT
#define complexFormat cufftComplex

#if Bits==12
using pixeltype=unsigned short;
#elif Bits==16
using pixeltype=unsigned short;
#else
using pixeltype=uchar;
#endif

#if Bits==12 || Bits==16
#define format_cv CV_16UC1
#else
#define format_cv CV_8UC1
#endif

using Real=float;
#define REALIDX FLOAT;
#define COMPLEXIDX FLOAT2;
using fftw_format=std::complex<Real>;
const int rcolor = pow(2,Bits);
#endif
#define sq(x) ((x)*(x))
#define sqSum(x,y) ((x)*(x)+(y)*(y))
