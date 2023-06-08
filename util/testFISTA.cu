#include <complex>
#include <cassert>
#include <stdio.h>
#include <time.h>
#include <random>

#include <stdio.h>
#include <libconfig.h++>
#include "cufft.h"
#include "common.h"
#include <ctime>
#include "cudaConfig.h"
#include "experimentConfig.h"
#include "cuPlotter.h"
#include "mnistData.h"
#include "tvFilter.h"

#include "cdi.h"

void applyC(Real* input, Real* output){
  cudaMemcpy(output, input, memMngr.getSize(input), cudaMemcpyDeviceToDevice);
}

int main(int argc, char** argv )
{
  CDI setups(argv[1]);
  if(argc < 2){
    printf("please feed the object intensity and phase image\n");
  }
  setups.readFiles();
  setups.init();
  setups.prepareIter();
  FISTA(setups.patternData,setups.patternData, 0.003, 100, 0);
  plt.plotFloat(setups.patternData, MOD, 1, setups.exposure, "test",1);
  return 0;
}

