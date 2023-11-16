#ifndef __COMMON_H__
#define __COMMON_H__
#include "format.h"
#include "stdlib.h"

// Declare the variables

#ifdef __cplusplus
extern "C" {
#endif
void writeComplexImage(const char* name, void* data, int row, int column);
void writeFloatImage(const char* name, void* data, int row, int col);
Real *readImage_c(const char* name, int *row, int *col, void* funcptr);
int writePng(const char* png_file_name, void* pix , int width, int height, int bit_depth, char colored);
int put_formula(const char* formula, int x, int y, int width, void* data, char iscolor, char rgb[3]);
void plotPng(const char* label, Real* data, char* cache, int rows, int cols, char iscolor);
void cvtLog(Real* data, int nele);
#ifdef __cplusplus
}
Real *readImage(const char* name, int &row, int &col, void* (cmalloc)(size_t) = 0);
#endif

#endif
