#include "cdilmdb.hpp"
#include "fmt/core.h"
#include <stdio.h>
#include <math.h>
const int N = 256*256;
const int M = 256*256;
int main(){
  int handle;
  initLMDB(&handle, "testdb");
  float data[N];
  float label[M];
  size_t sizes[2] = {N*sizeof(float),M*sizeof(float)};
  void** aa = (void**)malloc(2*sizeof(void*));
  aa[0] = (void*)data;
  aa[1] = (void*)label;
  for(int key = 0; key < 10; key++){
    for(int n = 0; n < N; n++){
      data[n] = sin((float)(n+key)/(60))*n/2;
    }
    for(int m = 0; m < M; m++){
      label[m] =cos((float)(m+key)/(60))*m;
    }
    fillLMDB(handle, &key,2,aa,sizes);
  }
  saveLMDB(handle);
  int key  = 0;
  size_t* data_sizes;
  float** dataout;
  int ndata = 0;
  readLMDB(handle, &ndata, (void***)&dataout, &data_sizes, &key);
  fmt::println("data= {:f}", dataout[0][10]);
  fmt::println("label= {:f}", dataout[1][10]);
  return 0;
}

