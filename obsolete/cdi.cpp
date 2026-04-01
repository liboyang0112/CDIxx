#include <unistd.h>
#include "fmt/core.h"
#include "imgio.hpp"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cdi.hpp"
#include <bits/stdc++.h>
//#include <tracy/Tracy.hpp>
using namespace std;
int main(int argc, char** argv )
{
  //ZoneScoped;
  CDI setups(argv[1]);
  if(argc < 2){
    fmt::println("please feed the object intensity and phase image");
  }
  setups.readFiles();
  setups.init();

  //-----------------------configure experiment setups-----------------------------
  fmt::println("Imaging distance = {:4.2f}cm", setups.d*1e-4);
  fmt::println("pupil Imaging distance = {:4.2f}cm", setups.dpupil*1e-4);
  Real fresnelNumber = M_PI*sq(setups.beamspotsize)/(setups.d*setups.lambda);
  fmt::println("Fresnel Number = {:f}",fresnelNumber);
  int sz = setups.row*setups.column*sizeof(complexFormat);
  complexFormat* cuda_pupilAmp = 0, *cuda_ESW = 0, *cuda_ESWP = 0, *cuda_ESWPattern = 0, *cuda_pupilAmp_SIM = 0;
  if(setups.dopupil){
    cuda_pupilAmp = (complexFormat*)memMngr.borrowCache(sz);
    if(setups.runSim) myMemcpyD2D(cuda_pupilAmp, setups.objectWave, sz);
  }
    if(setups.runSim && setups.domnist){
      for(int i = 0; i < setups.mnistN; i++){
        setups.save_suffix = to_string(i);
        setups.prepareIter();
        if(setups.doIteration) setups.phaseRetrieve();
      }
    }else{
      setups.prepareIter();
      if(setups.doIteration){
        setups.phaseRetrieve();
        setups.saveState();
      }
      double smallresidual = setups.residual;
      for(int i = 0; i < setups.nIter; i++){
        setups.prepareIter();
        if(setups.doIteration){
          setups.phaseRetrieve();
          if(smallresidual > setups.residual){
            smallresidual = setups.residual;
            setups.saveState();
          }
        }
      }
    }
  setups.checkAutoCorrelation();
  //Now let's do pupil
  return 0;
}

