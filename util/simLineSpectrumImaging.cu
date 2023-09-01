#include "cudaConfig.h"
#include "material.h"
#include "cdi.h"

int main(int argc, char* argv[]){
  ToyMaterial mat;
  CDI cdi(argv[1]);
  cdi.readFiles();
  cdi.init();
  int nlambda = 100;
  int row = cdi.row;
  int column = cdi.column;
  complexFormat** objectWaves = (complexFormat**)ccmemMngr.borrowCache(nlambda*sizeof(complexFormat*));
  Real* refIntensity = (Real*)ccmemMngr.borrowCache(nlambda*sizeof(Real));
  Real* refImg = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  for(int i = 1; i < nlambda ; i++){
    objectWaves[i] = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
    refIntensity[i] = 1;
  }
  objectWaves[0] = (complexFormat*)cdi.objectWave;
  refIntensity[0] = 1;
}
