#include "fmt/core.h"
#include <ctime>
#include "cudaConfig.hpp"
#include "fmt/core.h"
float bandwidthtest(int GB){
  size_t sz = 1024*1024;
  sz*= 1024*GB;
  void* memory = malloc(sz);
  myCuDMalloc(char, memorydev, sz);
  time_t current_time = time(NULL);
  myMemcpyH2D(memorydev, memory, sz);
  myMemcpyD2H(memory, memorydev, sz);
  myMemcpyH2D(memorydev, memory, sz);
  myMemcpyD2H(memory, memorydev, sz);
  myMemcpyH2D(memorydev, memory, sz);
  myMemcpyD2H(memory, memorydev, sz);
  fmt::println("time eclipsed: {}", time(NULL)-current_time);
  return 6.*GB/(time(NULL) - current_time);
}

int main(){
  float speed = bandwidthtest(3);
  fmt::println("CPU-GPU Bandwidth = {:4.2f} GB/s", speed);
  return 0;
}
