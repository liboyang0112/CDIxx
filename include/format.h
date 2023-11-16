#ifndef __FORMAT_H__
#define __FORMAT_H__
#define Bits 16
#include <complex.h>

#define VTK_TYPE VTK_FLOAT
//#define complexFormat float _Complex
#define complexFormat cufftComplex

#if Bits==12 || Bits==16
typedef  unsigned short pixeltype;
#else
typedef uchar pixeltype;
#endif

#if Bits==12 || Bits==16
#define format_cv CV_16UC1
#else
#define format_cv CV_8UC1
#endif

const int rcolor=65535;

#define Real float
#define REALIDX FLOAT
#define COMPLEXIDX FLOAT2
#endif
#define sq(x) ((x)*(x))
#define sqSum(x,y) ((x)*(x)+(y)*(y))
