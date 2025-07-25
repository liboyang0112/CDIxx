#include "cudaConfig.hpp"
#include "imgio.hpp"
#include "material.hpp"
#include "cub_wrap.hpp"
#include "memManager.hpp"
#include "spectImaging.hpp"
#include "cuPlotter.hpp"
#include <math.h>
#include <string>

//int main(int argc, char* argv[]){
  //if(argc < 2){
  //  printf("Usage: simLineSpectrumImaging_run xxx.cfg\n");
  //  return 0;
  //}
int main(){
  int row = 1024;
  int col = 1024;
  int mrow = 256;
  int mcol = 256;
  init_cuda_image();
  resize_cuda_image(mrow, mcol);
  const int nlambda = 7;
  double lambdas[nlambda] = {1, 1.1, 11./9, 11./8, 11./7, 11./6, 11./5};
  double spectra[nlambda] = {0.2,0.2,0.2,0.2,0.3,0.3,0.3};
  spectImaging workspace;
  workspace.init(row, col, nlambda, lambdas, spectra);
  const int ncenters = 2;
  const int refsize = 0;
  int xcenters[] = {(mrow>>1)+refsize, row-(mrow>>1)-refsize,(mrow>>1)+refsize,row-(mrow>>1)-refsize};
  int ycenters[] = {(mcol>>1)+refsize, (mcol>>1)+refsize,col - (mcol>>1) - refsize, col - (mcol>>1) - refsize};
  const int npoints = ncenters*(refsize*2+1)*(refsize*2+1);
  int xs[npoints];
  int ys[npoints];
  int idx = 0;
  for (int ic = 0 ; ic < ncenters; ic++) {
    for (int x = -refsize; x < refsize+1; x++) {
      for (int y = -refsize; y < refsize+1; y++) {
        xs[idx] = xcenters[ic] + x;
        ys[idx] = ycenters[ic] + y;
        idx++;
      }
    }
  }
  diamond m;
  m.startx = m.starty = 0;
  m.width = mrow;
  m.height = mcol;
  myCuDMalloc(Real, mask, mrow*mcol);
  myCuDMalloc(Real, support, mrow*mcol);
  createMask(support , &m);
  void* seed = newRand(mrow*mcol);
  initRand(seed, 1);
  randZero(support, mask, seed, 0.5, 1);
  plt.init(mrow, mcol);
  plt.plotFloat(mask, MOD, 0, 1, "mask");
  workspace.pointRefs(npoints, xs, ys);
  workspace.initHSI(mrow,mcol);
  //first, lets load data of indian panes.
  Real* imgcache = 0;
  Real* imgcache1 = 0;
  complexFormat* wave = 0;
  for (int i = 0; i < nlambda; i++) {
    int imgrow,imgcol;
    Real* intensity = readImage(("slice/out_" + std::to_string(i) + ".png").c_str(), imgrow, imgcol);
    Real* phase = readImage(("slice/out_" + std::to_string(i+nlambda) + ".png").c_str(), imgrow, imgcol);
    if(!imgcache) myCuMalloc(Real, imgcache, imgrow*imgcol);
    if(!imgcache1) myCuMalloc(Real, imgcache1, imgrow*imgcol);
    myMemcpyH2D(imgcache, intensity, imgrow*imgcol*sizeof(Real));
    myMemcpyH2D(imgcache1, phase, imgrow*imgcol*sizeof(Real));
    myFree(intensity);
    myFree(phase);
    resize_cuda_image(imgrow, imgcol);
    if(!wave) myCuMalloc(complexFormat, wave, imgrow*imgcol);
    createWaveFront(imgcache, imgcache1, wave, 1);
    resize_cuda_image(mrow, mcol);
    resize(wave, (complexFormat*)workspace.spectImages[i],imgrow, imgcol);
    multiply((complexFormat*)workspace.spectImages[i], (complexFormat*)workspace.spectImages[i], support);
    plt.plotComplexColor(workspace.spectImages[i], 0, 1, ("hsi_orig" + std::to_string(i)).c_str(), 0);
    multiply((complexFormat*)workspace.spectImages[i], (complexFormat*)workspace.spectImages[i], mask);
  }
  myCuDMalloc(Real, d_patternSum, row*col);
  workspace.generateMWLPattern(d_patternSum, 1);
  void* state = newRand(row*col);
  initRand(state, 0);
  resize_cuda_image(row, col);
  //ccdRecord((Real*)d_patternSum, (Real*)d_patternSum, 1, state, 0.02);
  plt.plotFloat(d_patternSum, MOD, 0, 1, "mergedlog", 1, 0, 1);
  plt.plotFloat(d_patternSum, MOD, 0, 1, "merged", 0, 0, 0);
  //applyNorm((Real*)d_patternSum, 50);
  workspace.plotAutoCorr("hologram", (Real*)d_patternSum, 1.);
  workspace.clearHSI();
  workspace.reconstructHSI(d_patternSum, mask);
  workspace.saveHSI("hsi_recon", support);
}
