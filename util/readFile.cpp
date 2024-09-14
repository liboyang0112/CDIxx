#include "imageFile.hpp"
#include <stdio.h>


int main(int argc, char* argv[]){
  if(argc < 2){
    printf("Usage: readFile_run imgfile.ext");
    return 0;
  }
  FILE* fin = fopen(argv[1], "r");
  imageFile f;
  if(!fread(&f, sizeof(f), 1, fin)){
    printf("WARNING: file %s is empty!\n", argv[1]);
  }
  printf("read type:%d, rows=%d, cols=%d\n", f.type, f.rows, f.cols);
  fclose(fin);
}
