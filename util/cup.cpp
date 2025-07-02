#include "imgio.hpp"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "misc.hpp"
#include "ptycho.hpp"
#include "tvFilter.hpp"
#include <cmath>
#include <string>

int main(){
  //Initialize CUP
  bool runSim = 1;
  int rows = 150, cols = 150, step_size = 10, ncomp=12;
  myDMalloc(int, shift, ncomp);
  for(int i = 0; i < ncomp; i++){
    shift[i] = step_size*ncomp;
  }
  Real* image;
  if(runSim) {
    image = readImage("einstein.png", rows, cols);
  }
  int colsc = cols+(ncomp-1)*step_size;
  int sz = rows*cols*sizeof(Real);
  myDMalloc(Real*, comps, ncomp);
  init_cuda_image();
  resize_cuda_image(rows,cols);
  for(int i = 0; i < ncomp; i++)
    myCuMalloc(Real, comps[i], rows*cols);
  myCuDMalloc(Real, compressed, rows*colsc);
  myCuDMalloc(char, mask, rows*rows);
  myCuDMalloc(Real, cache, rows*cols);
  void* devstate = newRand(rows*cols);
  initRand(devstate, 0);
  if(runSim){
    randMask(mask,devstate,.3);
    myMemcpyH2D(comps[0], image, sz);
    applyMask(comps[0], mask);
    plt.init(rows, cols);
    plt.plotFloat(comps[0], MOD, 0, 1, "masked", 0, 0, 0);
    for(int i = 0; i < ncomp; i++){
      paste(compressed, comps[0], colsc, 0, i*step_size, 0, (1 - Real(abs(ncomp-i*2-1))/ncomp)/3);
    }
    clearCuMem(comps[0], sz);
  }
  resize_cuda_image(rows,colsc);
  plt.init(rows, colsc);
  plt.plotFloat(compressed, MOD, 0, 1, "compressed", 0, 0, 0);
  //Start recovery
  int niter = 300;
  Real* residual = (Real*)memMngr.borrowSame(compressed);// y-Ax
  Real lambda = 1./ncomp;
  resize_cuda_image(rows,cols);
  for(int iter = 0; iter < niter ; iter ++){
    // Compute residual E = y-Ax
    myMemcpyD2D(residual, compressed, rows*colsc*sizeof(Real));
    for(int i = 0; i < ncomp; i++){
      myMemcpyD2D(cache, comps[i], sz);
      applyMask(cache, mask);
      paste(residual, cache, colsc, 0, i*step_size, 0, -1);
    }
    // Gradient G = 2*A^T*E = 2*crop[E], A is identity matrix in the window!
    // Update x = x - lambda * G
    for(int i = 0; i < ncomp; i++){
    //getWindow(T* object, int shiftx, int shifty, int objrow, int objcol, T *window, bool replace, Real norm);
      getWindow(residual, 0, i*step_size, rows, colsc, cache, 1, 2*lambda);
      applyMask(cache, mask);
      add(comps[i], cache);
      // Apply FISTA
      // FISTA(Real* b, Real* output, Real lambda, int niter, void (applyC)(Real*, Real*)){
      if(iter!=niter-1) FISTA(comps[i], comps[i], 1e-3, 1, 0);
    }
  }
  resize_cuda_image(rows,cols);
  plt.init(rows, cols);
  for(int i = 0; i < ncomp; i++) plt.plotFloat(comps[i], MOD, 0, 3, ("recon" + std::to_string(i)).c_str(), 0, 0, 0);
  resize_cuda_image(rows,colsc);
  plt.init(rows, colsc);
  plt.plotFloat(residual, MOD, 0, 1, "residual", 1, 0, 1);
  return 0;
}
