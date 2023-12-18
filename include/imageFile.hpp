#ifndef _IMAGEFILE_H_
#define _IMAGEFILE_H_
#define EMPTY 0
#define INT_8 1
#define INT_16 2
#define FLOAT 3
#define FLOAT2 4
#define DOUBLE 5
#define DOUBLE2 6
const int typeSizes[] = {0, 1, 2, 4, 8, 8, 16};
struct imageFile{
  char type;  // 8bit, 16bit, float, double
  int rows;
  int cols;
};
#endif
