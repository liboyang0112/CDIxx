#include <stdio.h>
#include <ctime>
float bandwidthtest(int GB){
  size_t sz = 1024*1024;
  sz*= 1024*GB;
  void* memory = malloc(sz); //3G
  void* memorydev;
  cudaMalloc(&memorydev, sz); //3G
  time_t current_time = time(NULL);
  cudaMemcpy(memorydev, memory, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(memory, memorydev, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(memorydev, memory, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(memory, memorydev, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(memorydev, memory, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(memory, memorydev, sz, cudaMemcpyDeviceToHost);
  printf("time eclipsed: %ld\n", time(NULL)-current_time);
  return 6.*GB/(time(NULL) - current_time);
}
