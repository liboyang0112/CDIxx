#include "cuPlotter.h"
#include "cudaDefs.h"
#include "cudaConfig.h"
#include <cassert>
cuPlotter plt;

void cuPlotter::freeCuda(){
  if(cuCache_data) { memMngr.returnCache(cuCache_data); cuCache_data = 0;}
  if(cuCache_float_data) { memMngr.returnCache(cuCache_float_data); cuCache_float_data = 0;}
}
__device__ Real cugetVal(cudaVars*vars, mode m, cuComplex &data, Real decay, bool islog){
  Real target = 0;
  if(m==REAL) {
    target = data.x*decay;
    if(islog){
      if(target!=0){
        Real ab = fabs(target);
        Real logv = log2f(ab)/log2f(vars->rcolor)+1;
        if(logv < 0) target = 0;
        else target = target*logv/(2*ab);
      }
    }
    return (target+0.5)*vars->rcolor;
  }
  if(m==IMAG) target = data.y*decay;
  if(m==MOD) target = cuCabsf(data)*decay;
  if(m==MOD2) target = (data.x*data.x+data.y*data.y)*decay;
  if(m==PHASE){
    target = atan2(data.y,data.x)/2/M_PI+0.5;
  }
  if(m==PHASERAD){
    target = atan2(data.y,data.x);
  }
  if(islog){
    if(target!=0)
      target = log2f(target)/log2f(vars->rcolor)+1;
  }
  target*=vars->rcolor;
  return target;
}
__device__ Real cugetVal(cudaVars* vars, mode m, Real &data, Real decay, bool islog){
  Real ret = 0;
  if(m==REAL) {
    ret = data*decay; //-1~1
    if(islog){
      if(ret!=0){
        Real ab = fabs(ret);
        Real logv = log2f(ab)/log2f(vars->rcolor)+1;
        if(logv < 0) ret = 0;
        else ret = ret*logv/(2*ab);
      }
    }
    return (ret+0.5)*vars->rcolor;
  }
  if(m==MOD2) ret = data*data*decay;
  if(m==MOD) ret = fabs(data)*decay;
  if(islog){
    if(ret!=0)
      ret = log2f(ret)/log2f(vars->rcolor)+1;
  }
  ret*=vars->rcolor;
  return ret;
}

template <typename T>
__global__ void process(cudaVars* vars, int cuda_row, int cuda_column, void* cudaData, pixeltype* cache, mode m, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0){
  cudaIdx()
  int halfrow = cuda_row>>1;
  int halfcol = cuda_column>>1;
  int targetx = x;
  int targety = y;
  if(isFrequency) {
    targetx = x<halfrow?x+halfrow:(x-halfrow);
    targety = y<halfcol?y+halfcol:(y-halfcol);
  }
  if(isFlip){
    targetx = cuda_row-x-1;
  }
  T data = ((T*)cudaData)[index];
  Real target = cugetVal(vars,m,data,decay,islog);
  if(target < 0) target = 0;
  if(target>=vars->rcolor) {
    target=vars->rcolor-1;
  }
  if(target!=target) {
 //   printf("ERROR: target is NAN\n");
 //   exit(0);
  }
  cache[targetx*cuda_column+targety] = floor(target);
}
__global__ void getPhase(cudaVars* vars, int cuda_row, int cuda_column, void* cudaData, Real* cache, mode m, bool isFrequency=0, Real decay = 1, bool isFlip = 0){
  cudaIdx()
  int halfrow = cuda_row>>1;
  int halfcol = cuda_column>>1;
  int targetx = x;
  int targety = y;
  if(isFrequency) {
    targetx = x<halfrow?x+halfrow:(x-halfrow);
    targety = y<halfcol?y+halfcol:(y-halfcol);
  }
  if(isFlip){
    targetx = cuda_row-x;
  }
  cache[targetx*cuda_column+targety] =cugetVal(vars, m,((cuComplex*)cudaData)[index],decay,0);
}

void* cuPlotter::processPhaseData(void* cudaData, const mode m, bool isFrequency, Real decay, bool isFlip){
  if(!cuCache_float_data) cuCache_float_data = (Real*) memMngr.borrowCache(rows*cols*sizeof(Real));
  getPhase<<<numBlocks,threadsPerBlock>>>(cudaVar, rows, cols, cudaData, (Real*)cuCache_float_data, m, isFrequency, decay, isFlip);
  cudaMemcpy(cv_float_data, cuCache_float_data,rows*cols*sizeof(Real), cudaMemcpyDeviceToHost); 
  return cv_float_data;
};

void cuPlotter::saveFloatData(void* cudaData){
  cudaMemcpy(cv_float_data, cudaData, rows*cols*sizeof(Real), cudaMemcpyDeviceToHost); 
};
void cuPlotter::saveComplexData(void* cudaData){
  cudaMemcpy(cv_complex_data, cudaData, rows*cols*sizeof(complexFormat), cudaMemcpyDeviceToHost); 
};
void* cuPlotter::processFloatData(void* cudaData, const mode m, bool isFrequency, Real decay, bool islog, bool isFlip){
  if(!cuCache_data) cuCache_data = (pixeltype*) memMngr.borrowCache(rows*cols*sizeof(pixeltype));
  process<Real><<<numBlocks,threadsPerBlock>>>(cudaVar, rows, cols, cudaData, cuCache_data, m, isFrequency, decay, islog, isFlip);
  cudaMemcpy(cv_data, cuCache_data,rows*cols*sizeof(pixeltype), cudaMemcpyDeviceToHost); 
  return cv_data;
};
void* cuPlotter::processComplexData(void* cudaData, const mode m, bool isFrequency, Real decay, bool islog, bool isFlip){
  if(!cuCache_data) cuCache_data = (pixeltype*) memMngr.borrowCache(rows*cols*sizeof(pixeltype));
  process<cuComplex><<<numBlocks,threadsPerBlock>>>(cudaVar, rows, cols, cudaData, cuCache_data, m,isFrequency, decay, islog, isFlip);
  cudaMemcpy(cv_data, cuCache_data,rows*cols*sizeof(pixeltype), cudaMemcpyDeviceToHost); 
  return cv_data;
};
