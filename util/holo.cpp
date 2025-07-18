#include <stdio.h>

#include "fmt/core.h"
#include "holo.hpp"

int main(int argc, char** argv )
{
  holo setups(argv[1]);
  if(argc < 2){
    fmt::println("please feed the object intensity and phase image");
  }
  setups.simulate();
  return 0;
}

