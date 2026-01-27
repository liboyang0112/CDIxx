#include <string>
#include <cstring>
#include <math.h>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "fmt/core.h"
using namespace std;
void initV(Real* V, Real val);
void Hpsifunc (Real * psi, Real *V, Real *Hpsi, Real Eshift);

int main(){
  init_cuda_image();
  int nsteps = 1000;
  const int nx = 200;
  const int ny = 200;
  const int nz = 200;

  size_t nnode = nx*ny*nz;
  myCuDMalloc(Real, psi, nnode);
  myCuDMalloc(Real, Hpsi, nnode);
  myCuDMalloc(Real, H2psi, nnode);
  myCuDMallocClean(Real, mom, nnode);
  myCuDMalloc(Real, V, nnode);
  //select slice for visualization
  myCuDMalloc(Real, slice, nx*ny);
  Real beta = 0.1;
  resize_cuda_image(nx,ny,nz);
  initV(V, -0.2);
  initV(H2psi, 1);
  Real sum = findRootSumSq(H2psi);
  fmt::println("sum = {:f}", sum);
  applyNorm(H2psi, 1./sum);

  plt.init(nx,ny);
  int psivid = plt.initVideo("psi.mp4", 24);
  for(int i = 0; i < nsteps; i++){
    resize_cuda_image(nx,ny,nz);
    add(psi, H2psi, mom, 1-beta);
    if(i > 100) beta = 0.01;
    Hpsifunc(psi, V, Hpsi, -12);
    myMemcpyD2D(psi,H2psi,nx*ny*nz*sizeof(Real));
    Hpsifunc(Hpsi, V, H2psi, -12);
    applyNorm(H2psi, 1./findRootSumSq(H2psi));
    add(mom, H2psi, psi, -1);
    if(i%3==0) {
      resize_cuda_image(nx,ny);
      getXYSlice(slice, H2psi , nx, ny, nz/2);
      plt.toVideo = psivid;
      plt.plotFloat(slice, MOD2, 0, 5e4, "",1,0,1,("psi,iter="+to_string(i)).c_str());
      //applyNorm(slice, 1./findMax(slice));
      //plt.videoFrame(slice);
    }
  }
  plt.saveVideo(psivid);
  plt.toVideo = -1;
  plt.plotFloat(slice, MOD2, 0, 5e4, "psi",1,0,1);
}
