#include "cudaConfig.h"
#include "material.h"
#include "cub_wrap.h"
#include "cdi.h"
#include "common.h"

cuFunc(assignX, (Real* img, int* intimg),(img, intimg),{
  cuda1Idx();
  int x = index/cuda_column;
  if(img[index] > vars->threshold){
    intimg[index] = x;
  }
})

cuFunc(assignY, (Real* img, int* intimg),(img, intimg),{
  cuda1Idx();
  int y = index%cuda_column;
  if(img[index] > vars->threshold){
    intimg[index] = y;
  }
})

int main(int argc, char* argv[]){
  ToyMaterial mat;
  CDI cdi(argv[1]);
  cdi.readFiles();
  cdi.init();
  cdi.prepareIter();
  if(cdi.doIteration){
    cdi.phaseRetrieve();
    cdi.saveState();
  } // Inside CDI class is the object support and reference image, which was measured without sample.
  int row = cdi.row;
  int column = cdi.column;
  //split reference and object support into two images.
  int nlambda = 100;
  Real* refSpectrum = (Real*)ccmemMngr.borrowCache(nlambda*sizeof(Real));
  int mrow, mcol;
  Real* refMask = readImage(cdi.pupil.Intensity, mrow, mcol);
  int pixCount = 0;
  for(int idx = 0; idx < mrow*mcol ; idx++){
    if(refMask[idx] > 0.5) pixCount++;
  }
  int2* maskMap = (int2*)memMngr.borrowCache(pixCount*sizeof(int2));
  Real* objSupport = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  //since the support region is much smaller than the full image. we save the exit wave in smaller caches.
  int border = 5;
  int* tmp = (int*)memMngr.borrowCache(row*column*sizeof(int));
  assignX(objSupport, tmp);
  int2 supportxrange = {findMin(tmp)-border, findMax(tmp)+border};
  assignY(objSupport, tmp);
  int2 supportyrange = {findMin(tmp)-border, findMax(tmp)+border};
  int2 supportdim = {supportxrange.y-supportxrange.x, supportyrange.y-supportyrange.x};
  complexFormat** objectWaves = (complexFormat**)ccmemMngr.borrowCache(nlambda*sizeof(complexFormat*));
  for(int i = 0; i < nlambda ; i++){
    refSpectrum[i] = 1;
    objectWaves[i] = (complexFormat*)memMngr.borrowCache(supportdim.x*supportdim.y*sizeof(complexFormat));
  }
}
