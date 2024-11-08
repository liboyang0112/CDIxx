extern "C" {
  #include "freetype.hpp"
  #include <freetype/freetype.h>
}
#include "imgio.hpp"
#include <stdint.h>
#include "memManager.hpp"
int main( int argc, char**  argv )
{
  int rows = 150;
  int cols = 150;
  myDMalloc(uint16_t, data, rows*cols);
  memset(data, 0, rows*cols*sizeof(uint16_t));
  unsigned char val[3] = {0xff, 0xff, 0xff};
  if(argc < 2) {
    printf("Usage: freetype_run text\n");
    return 0;
  }
  putText(argv[1], 0, rows, rows, cols, data, 0, val);
  writePng("test.png", data, rows, cols, 16, 0);
  return 0;
}
