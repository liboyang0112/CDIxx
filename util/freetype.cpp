extern "C" {
  #include "freetype.h"
  #include <freetype/freetype.h>
}
#include <math.h>
#include "imgio.h"
#include <stdint.h>
#include "memManager.h"
int main( int argc, char**  argv )
{
  int rows = 150;
  int cols = 150;
  myDMalloc(uint16_t, data, rows*cols);
  memset(data, 0, rows*cols*sizeof(uint16_t));
  unsigned char val[3] = {0xff, 0xff, 0xff};
  putText("wss", 0, rows, rows, cols, data, 0, val);
  writePng("test.png", data, rows, cols, 16, 0);
  return 0;
}
