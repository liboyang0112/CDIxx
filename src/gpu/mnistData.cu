#include "mnistData.h"
#include "cudaDefs.h"
#include "cudaConfig.h"
#include "imgio.h"
#include "cuPlotter.h"

cuMnist::cuMnist(const char* dir, int nm, int re, int r, int c) : mnistData(dir), refinement(re), row(r), col(c), nmerge(nm){
  cuRaw = memMngr.borrowCache(rowraw*colraw*sizeof(Real));
  rowrf = rowraw*nmerge;
  colrf = colraw*nmerge;
  cuRefine = memMngr.borrowCache(rowrf*colrf*refinement*refinement*sizeof(Real));
  if(refinement!=1){
    createPlan(&handle, rowrf*refinement, colrf*refinement);
    createPlan(&handleraw, rowrf, colrf);
  }
  myCuMalloc(complexFormat, cacheraw, rowrf*colrf);
  myCuMalloc(complexFormat, cache, rowrf*colrf*refinement*refinement);
};

cuFunc(paste, (Real* out, Real* in, int rowin, int colin, int posx, int posy),(out, in, rowin, colin, posx, posy),{
  cudaIdx();
  if(x < posx || y < posy) return;
  if(x >= posx+rowin || y > posy+colin) return;
  int xin = x - posx;
  int yin = y - posy;
  Real data = in[xin*colin+yin];
  if(data < 0.5) data = 0;
  else data = 1;
  out[index] += data;
  if(out[index] > 1) out[index] = 1;
})
void cuMnist::cuRead(void* out){
  Real val = nmerge;
  resize_cuda_image(rowrf, colrf);
  init_cuda_image(65536,1);
  //cudaMemcpy(cuRaw, read(), rowraw*colraw*sizeof(Real), cudaMemcpyHostToDevice);
  //refine((Real*)cuRaw, (Real*)cuRefine, refinement);
  auto media = (refinement==1?cuRefine:out);
  cudaMemset(media,0,rowrf*colrf*sizeof(Real));
  for(int i = 0; i < val; i++){
    for(int j = 0; j < val; j++){
      cudaMemcpy(cuRaw, read(), rowraw*colraw*sizeof(Real), cudaMemcpyHostToDevice);
      paste( (Real*)media, (Real*)cuRaw, rowraw, colraw, rowraw*i*2/3, colraw*j*2/3);
    }
  }
  if(refinement!=1){
    resize_cuda_image(rowrf, colrf);
    extendToComplex((Real*)out, cacheraw);
    myFFTM(handleraw,cacheraw, cacheraw);
    resize_cuda_image(rowrf*refinement, colrf*refinement);
    padinner(cacheraw, cache, rowrf, colrf, 1./(rowrf*colrf));
    myIFFTM(handle,cache, cache);
    getMod((Real*)cache, cache);
    applyThreshold((Real*)cache, (Real*)cache, 0.5);
    resize_cuda_image(row, col);
    pad( (Real*)cache, (Real*)out,rowrf*refinement, colrf*refinement);
  }else{
    resize_cuda_image(row, col);
    pad( (Real*)media, (Real*)out,rowrf, colrf);
  }
  gpuErrchk(cudaGetLastError());
}
