#include "fmt/core.h"
#include "readCXI.hpp"
#include <stdlib.h>
int main(int argc, char* argv[]){
  if(argc < 2) fmt::println("please feed the CXI file and index of the entry"); 
  Real ** mask = new Real*();
  readCXI (argv[1], atoi(argv[2]), mask);
  return 0;
}
