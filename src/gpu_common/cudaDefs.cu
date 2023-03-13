#include "cudaDefs.h"
cudaVars* cudaVar = 0;
cudaVars* cudaVarLocal = 0;
dim3 numBlocks;
const dim3 threadsPerBlock(16,16);
complexFormat *cudaData = 0;
cufftHandle *plan, *planR2C;
void cuMemManager::c_malloc(void*& ptr, size_t sz) { gpuErrchk(cudaMalloc((void**)&ptr, sz)); }
cuMemManager memMngr;
void init_cuda_image(int rows, int cols, int rcolor, Real scale){
  const int sz = sizeof(cudaVars);
  if(!cudaVar){
    cudaMalloc((void**)&cudaVar, sz);
    cudaVarLocal = (cudaVars*) malloc(sz);
    cudaVarLocal->rows = rows;
    cudaVarLocal->cols = cols;
    cudaVarLocal->threshold = 0.5;
    cudaVarLocal->beta_HIO = 0.9;
    numBlocks.x=(rows-1)/threadsPerBlock.x+1;
    numBlocks.y=(cols-1)/threadsPerBlock.y+1;
    if(scale==scale) cudaVarLocal->scale = scale;
    if(rcolor!=0) cudaVarLocal->rcolor = rcolor;
    cudaMemcpy(cudaVar, cudaVarLocal, sz, cudaMemcpyHostToDevice);
    return;
  }
  if(cudaVarLocal->rows!=rows){
    cudaVarLocal->rows = rows;
    numBlocks.x=(rows-1)/threadsPerBlock.x+1;
  }
  if(cudaVarLocal->cols!=cols){
    cudaVarLocal->cols = cols;
    numBlocks.y=(cols-1)/threadsPerBlock.y+1;
  }
  if(rcolor!=0) cudaVarLocal->rcolor = rcolor;
  if(scale==scale) cudaVarLocal->scale = scale;
  cudaMemcpy(cudaVar, cudaVarLocal, sz, cudaMemcpyHostToDevice);
};
