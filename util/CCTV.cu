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

#include "CCTV.h"

int main(int argc, char** argv )
{
  CCTV setups(argv[1]);
  cudaFree(0); // to speed up the cudaMalloc; https://forums.developer.nvidia.com/t/cudamalloc-slow/40238
  if(argc < 2){
    printf("please feed the object intensity and phase image\n");
  }
  setups.readFiles();
  setups.init();

  //-----------------------configure experiment setups-----------------------------
  printf("Imaging distance = %4.2fcm\n", setups.d*1e-4);
  printf("forward norm = %f\n", setups.forwardFactor);
  printf("backward norm = %f\n", setups.inverseFactor);
  printf("fresnel factor = %f\n", setups.fresnelFactor);
  printf("enhancement = %f\n", setups.enhancement);

  printf("pupil Imaging distance = %4.2fcm\n", setups.dpupil*1e-4);
  printf("pupil forward norm = %f\n", setups.forwardFactorpupil);
  printf("pupil backward norm = %f\n", setups.inverseFactorpupil);
  printf("pupil fresnel factor = %f\n", setups.fresnelFactorpupil);
  printf("pupil enhancement = %f\n", setups.enhancementpupil);

  Real fresnelNumber = M_PI*pow(setups.beamspotsize,2)/(setups.d*setups.lambda);
  printf("Fresnel Number = %f\n",fresnelNumber);

  if(setups.doIteration) {
    if(setups.runSim && setups.domnist){
      for(int i = 0; i < setups.mnistN; i++){
        setups.save_suffix = to_string(i);
        setups.prepareIter();
        setups.phaseRetrieve(); 
      }
    }else{
      setups.prepareIter();
      setups.phaseRetrieve(); 
    }
    setups.saveState();
  }
  return 0;
}

