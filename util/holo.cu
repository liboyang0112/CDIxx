#include <complex>
#include <cassert>
#include <stdio.h>
#include <time.h>
#include <random>

#include <stdio.h>
#include <libconfig.h++>
#include "cufft.h"
#include "imgio.h"
#include <ctime>
#include "cudaConfig.h"
#include "cuPlotter.h"
#include "holo.h"

int main(int argc, char** argv )
{
  holo setups(argv[1]);
  if(argc < 2){
    printf("please feed the object intensity and phase image\n");
  }
  setups.simulate();
  return 0;
}

