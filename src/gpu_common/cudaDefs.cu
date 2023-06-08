#include "cudaDefs.h"
cudaVars* cudaVar = 0;
cudaVars* cudaVarLocal = 0;
dim3 numBlocks;
const dim3 threadsPerBlock(16,16);
complexFormat *cudaData = 0;
cufftHandle *plan, *planR2C;
int2 cuda_imgsz = {0,0};
void cuMemManager::c_malloc(void*& ptr, size_t sz) { gpuErrchk(cudaMalloc((void**)&ptr, sz)); }
cuMemManager memMngr;
void resize_cuda_image(int rows, int cols){
  cuda_imgsz.x = rows;
  numBlocks.x=(rows-1)/threadsPerBlock.x+1;
  cuda_imgsz.y = cols;
  numBlocks.y=(cols-1)/threadsPerBlock.y+1;
}
void init_cuda_image(int rcolor, Real scale){
  const int sz = sizeof(cudaVars);
  if(!cudaVar){
    cudaVar = (cudaVars*) memMngr.borrowCache(sz);
    cudaVarLocal = (cudaVars*) malloc(sz);
    cudaVarLocal->threshold = 0.5;
    cudaVarLocal->beta_HIO = 0.9;
    if(scale==scale) cudaVarLocal->scale = scale;
    if(rcolor!=0) cudaVarLocal->rcolor = rcolor;
    else cudaVarLocal->rcolor=65535;
    cudaMemcpy(cudaVar, cudaVarLocal, sz, cudaMemcpyHostToDevice);
    return;
  }
  if(rcolor!=0) cudaVarLocal->rcolor = rcolor;
  if(scale==scale) cudaVarLocal->scale = scale;
  cudaMemcpy(cudaVar, cudaVarLocal, sz, cudaMemcpyHostToDevice);
};
