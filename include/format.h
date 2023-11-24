#ifndef __FORMAT_H__
#define __FORMAT_H__
#define Bits 16

#define VTK_TYPE VTK_FLOAT

#if Bits==12 || Bits==16
#define complexFormat float _Complex
typedef  unsigned short pixeltype;
#else
typedef uchar pixeltype;
#endif

const int rcolor=65535;

#define Real float
#define REALIDX FLOAT
#define COMPLEXIDX FLOAT2
#endif
#define sq(x) ((x)*(x))
#define sqSum(x,y) ((x)*(x)+(y)*(y))
