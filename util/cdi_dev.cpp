#include <stdio.h>
#include "cdi.h"
using namespace std;

int main(int argc, char** argv )
{
  CDI setups(argv[1]);
  setups.readFiles();
  setups.init();
  if(setups.doIteration) {
    if(setups.runSim && setups.domnist){
      for(int i = 0; i < setups.mnistN; i++){
        setups.save_suffix = to_string(i);
        setups.prepareIter();
        setups.phaseRetrievedev(); 
      }
    }else{
      setups.prepareIter();
      setups.phaseRetrievedev();
      setups.saveState();
      double smallresidual = setups.residual;
      for(int i = 0; i < setups.nIter; i++){
        setups.prepareIter();
        setups.phaseRetrievedev();
        if(smallresidual > setups.residual){
          smallresidual = setups.residual;
          setups.saveState();
        }
      }
    }
  }
  setups.checkAutoCorrelation();
  return 0;
}

