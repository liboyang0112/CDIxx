#ifndef __FORMAT_H__
#define __FORMAT_H__
#define CCDBits 16
#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif
#define VTK_TYPE VTK_FLOAT

#if CCDBits==12 || CCDBits==16
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

#define loop(var, max) for(int var = 0; var < max; var++)
#define Real float
#ifndef __CUDACC__
struct float2 {
  float x, y;
#ifdef __cplusplus
  float2() : x(0), y(0) {}
  float2(float x, float y) : x(x), y(y) {}
  float2(double v) : x((float)v), y(0) {}
#endif
};
#endif // __CUDACC__
#ifdef __cplusplus
#include <cmath>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
extern "C++" {
#ifndef __CUDACC__
HD inline float2 make_float2(float x, float y) { return {x, y}; }
#endif
HD inline float2 operator+(const float2& a, const float2& b) { return {a.x+b.x, a.y+b.y}; }
HD inline float2 operator-(const float2& a, const float2& b) { return {a.x-b.x, a.y-b.y}; }
HD inline float2 operator*(const float2& a, const float2& b) { return {a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x}; }
HD inline float2 operator*(const float2& a, float b) { return {a.x*b, a.y*b}; }
HD inline float2 operator*(float a, const float2& b) { return {a*b.x, a*b.y}; }
HD inline float2 operator/(const float2& a, const float2& b) { float d=b.x*b.x+b.y*b.y; return {(a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d}; }
HD inline float2 operator/(const float2& a, float b) { return {a.x/b, a.y/b}; }
HD inline float2& operator+=(float2& a, const float2& b) { a.x+=b.x; a.y+=b.y; return a; }
HD inline float2& operator-=(float2& a, const float2& b) { a.x-=b.x; a.y-=b.y; return a; }
HD inline float2& operator*=(float2& a, const float2& b) { float2 r = a*b; a=r; return a; }
HD inline float2& operator*=(float2& a, float b) { a.x*=b; a.y*=b; return a; }
HD inline float2& operator/=(float2& a, float b) { a.x/=b; a.y/=b; return a; }
HD inline float2 conj(const float2& a) { return {a.x, -a.y}; }
HD inline float crealf(float2 a) { return a.x; }
HD inline float cimagf(float2 a) { return a.y; }
HD inline float creal(float2 a) { return a.x; }
HD inline float cimag(float2 a) { return a.y; }
HD inline float cabsf(float2 a) { return sqrtf(a.x*a.x + a.y*a.y); }
HD inline float cabs(float2 a) { return sqrtf(a.x*a.x + a.y*a.y); }
HD inline float cargf(float2 a) { return atan2f(a.y, a.x); }
HD inline float carg(float2 a) { return atan2f(a.y, a.x); }
}
#undef HD
#endif // __cplusplus
#define complexFormat float2
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
