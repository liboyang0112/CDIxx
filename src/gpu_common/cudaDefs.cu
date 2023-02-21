#include "cudaDefs.h"
__device__ __constant__ Real cuda_beta_HIO;
__device__ __constant__ int cuda_row;
__device__ __constant__ int cuda_column;
__device__ __constant__ int cuda_rcolor;
__device__ __constant__ Real cuda_scale;
__device__ __constant__ int cuda_totalIntensity;
__device__ __constant__ Real cuda_threshold = 0.5;
dim3 numBlocks;
const dim3 threadsPerBlock(16,16);
complexFormat *cudaData = 0;
cufftHandle *plan, *planR2C;
void cuMemManager::c_malloc(void*& ptr, size_t sz) { gpuErrchk(cudaMalloc((void**)&ptr, sz)); }
cuMemManager memMngr;
static int rows_, cols_;
void init_cuda_image(int rows, int cols, int rcolor, Real scale){
  if(rows!=rows_ || cols!=cols_){
    cudaMemcpyToSymbol(cuda_row,&rows,sizeof(rows));
    cudaMemcpyToSymbol(cuda_column,&cols,sizeof(cols));
    numBlocks.x=(rows-1)/threadsPerBlock.x+1;
    numBlocks.y=(cols-1)/threadsPerBlock.y+1;
    rows_ = rows;
    cols_ = cols;
  }
  cudaMemcpyToSymbol(cuda_rcolor,&rcolor,sizeof(rcolor));
  cudaMemcpyToSymbol(cuda_scale,&scale,sizeof(scale));
};
