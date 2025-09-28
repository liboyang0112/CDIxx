#include <ctime>
#include "fmt/core.h"
float bandwidthtest(int GB){
  size_t sz = 1024*1024;
  sz*= 1024*GB;
  void* memory = malloc(sz);
  void* memorydev;
  cudaMalloc(&memorydev, sz);
  time_t current_time = time(NULL);
  cudaMemcpy(memorydev, memory, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(memory, memorydev, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(memorydev, memory, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(memory, memorydev, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(memorydev, memory, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(memory, memorydev, sz, cudaMemcpyDeviceToHost);
  fmt::println("time eclipsed: {}", time(NULL)-current_time);
  return 6.*GB/(time(NULL) - current_time);
}
