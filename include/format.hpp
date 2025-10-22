#ifndef __FORMAT_H__
#define __FORMAT_H__
#define Bits 16
#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif
#define VTK_TYPE VTK_FLOAT

#if Bits==12 || Bits==16
#define complexFormat float _Complex
typedef  unsigned short pixeltype;
#else
typedef uchar pixeltype;
#endif

typedef struct col_rgb{
    char r;
    char g;
    char b;
} col_rgb;

const int rcolor=65535;
enum Algorithm {RAAR, ER, POSER, HIO, POSHIO, FHIO, shrinkWrap, XCORRELATION, KKT, TV, cnt};

#define Real float
#define REALIDX FLOAT
#define COMPLEXIDX FLOAT2
#endif
#define sq(x) ((x)*(x))
#define sqSum(x,y) ((x)*(x)+(y)*(y))
#define SWAP(a, b) do { \
    __typeof__(a) temp = (a); \
    (a) = (b); \
    (b) = temp; \
} while(0)
