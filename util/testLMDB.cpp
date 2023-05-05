#include "cdilmdb.h"
#include <stdio.h>
#include <math.h>
const int N = 256*256;
const int M = 256*256;
int main(){
  initLMDB("testdb");
  float data[N];
  float label[M];
  for(int key = 0; key < 10; key++){
    for(int n = 0; n < N; n++){
      data[n] = sin((float)(n+key)/(60))*n/2;
    }
    for(int m = 0; m < M; m++){
      label[m] =cos((float)(m+key)/(60))*m;
    }
    fillLMDB(&key,(void*)data,N*sizeof(float),(void*)label, M*sizeof(float));
  }
  saveLMDB();
  int key  = 0;
  size_t datasz = N*sizeof(float);
  size_t labelsz = N*sizeof(float);
  float* dataout;
  float* labelout;
  readLMDB((void**)&dataout, &datasz, (void**)&labelout, &labelsz, &key);
  printf("data= %f\n", dataout[10]);
  printf("label= %f\n", labelout[10]);
  return 0;
}

