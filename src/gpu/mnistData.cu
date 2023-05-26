#include "mnistData.h"
#include "cudaConfig.h"
#include "common.h"
#include "cuPlotter.h"

cuMnist::cuMnist(const char* dir, int nm, int re, int r, int c) : mnistData(dir), refinement(re), row(r), col(c), nmerge(nm){
  cuRaw = memMngr.borrowCache(rowraw*colraw*sizeof(Real));
  rowrf = rowraw*nmerge;
  colrf = colraw*nmerge;
  cuRefine = memMngr.borrowCache(rowrf*colrf*refinement*refinement*sizeof(Real));
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
  init_cuda_image(rowrf, colrf, 65536, 1);
  //cudaMemcpy(cuRaw, read(), rowraw*colraw*sizeof(Real), cudaMemcpyHostToDevice);
  //cudaF(refine,(Real*)cuRaw, (Real*)cuRefine, refinement);
  auto media = (refinement==1?cuRefine:out);
  cudaMemset(media,0,rowrf*colrf*sizeof(Real));
  for(int i = 0; i < val; i++){
    for(int j = 0; j < val; j++){
      cudaMemcpy(cuRaw, read(), rowraw*colraw*sizeof(Real), cudaMemcpyHostToDevice);
      cudaF(paste, (Real*)media, (Real*)cuRaw, rowraw, colraw, rowraw*i*2/3, colraw*j*2/3);
    }
  }
  if(refinement!=1){
    init_cuda_image(rowrf*refinement, colrf*refinement, 65536, 1);
    cudaF(refine,(Real*)out, (Real*)cuRefine, refinement);
    init_cuda_image(row, col, 65536, 1);
    cudaF(pad, (Real*)cuRefine, (Real*)out,rowrf*refinement, colrf*refinement);
  }else{
    init_cuda_image(row, col, 65536, 1);
    cudaF(pad, (Real*)media, (Real*)out,rowrf, colrf);
  }
  gpuErrchk(cudaGetLastError());
}
