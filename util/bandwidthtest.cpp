#include "fmt/core.h"
float bandwidthtest(int GB);

int main(){
  float speed = bandwidthtest(3);
  fmt::println("CPU-GPU Bandwidth = {:4.2f} GB/s", speed);
  return 0;
}
