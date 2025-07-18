#include <stdio.h>
#include "cuPlotter.hpp"
#include "fmt/core.h"
#include "tvFilter.hpp"
#include "cdi.hpp"

int main(int argc, char** argv )
{
  CDI setups(argv[1]);
  if(argc < 2){
    fmt::println("please feed the object intensity and phase image");
  }
  setups.readFiles();
  setups.init();
  setups.prepareIter();
  FISTA(setups.patternData,setups.patternData, 0.003, 100, 0);
  plt.plotFloat(setups.patternData, MOD, 1, setups.exposure, "test",1);
  return 0;
}

