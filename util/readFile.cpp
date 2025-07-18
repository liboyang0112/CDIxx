#include "fmt/core.h"
#include "imageFile.hpp"
#include <stdio.h>


int main(int argc, char* argv[]){
  if(argc < 2){
    fmt::print("Usage: readFile_run imgfile.ext");
    return 0;
  }
  FILE* fin = fopen(argv[1], "r");
  imageFile f;
  if(!fread(&f, sizeof(f), 1, fin)){
    fmt::println("WARNING: file {} is empty!", argv[1]);
  }
  fmt::println("read type:{}, rows={}, cols={}", static_cast<signed char>(f.type), f.rows, f.cols);
  fclose(fin);
}
