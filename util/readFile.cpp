#include "common.h"
#include "imageFile.h"


int main(int argc, char* argv[]){
  FILE* fin = fopen(argv[1], "r");
  imageFile f;
  fread(&f, sizeof(f), 1, fin);
  printf("read type:%d, rows=%d, cols=%d\n", f.type, f.rows, f.cols);
  fclose(fin);
}
