#include <stdio.h>
float bandwidthtest(int GB);

int main(){
  float speed = bandwidthtest(3);
  printf("CPU-GPU Bandwidth = %4.2f GB/s\n", speed);
  return 0;
}
