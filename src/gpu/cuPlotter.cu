#include "format.hpp"
enum mode {MOD2,MOD, REAL, IMAG, PHASE, PHASERAD};
#include "cudaDefs_h.cu"
__forceinline__ __device__ void hsvToRGB(Real H, Real S, Real V, char* rgb){
    H*=6;
    char hi = floor(H);
    Real f = H - hi;
    unsigned char p = floor(V*(1-S)*255);
    unsigned char q = floor(V*(1-f*S)*255);
    unsigned char t = floor(V*(1-(1-f)*S)*255);
    unsigned char Vi = floor(V*255);
    switch(hi){
        case -1:
        case 0:
          rgb[0] = Vi;
          rgb[1] = t;
          rgb[2] = p;
          break;
        case 1:
          rgb[0] = q;
          rgb[1] = Vi;
          rgb[2] = p;
          break;
        case 2:
          rgb[0] = p;
          rgb[1] = Vi;
          rgb[2] = t;
          break;
        case 3:
          rgb[0] = p;
          rgb[1] = q;
          rgb[2] = Vi;
          break;
        case 4:
          rgb[0] = t;
          rgb[1] = p;
          rgb[2] = Vi;
          break;
        case 5:
        case 6:
          rgb[0] = Vi;
          rgb[1] = p;
          rgb[2] = q;
          break;
        default:
          printf("WARNING: HSV not recognized: %f, %f, %f, %d\n", H, S, V, hi);
          rgb[0] = rgb[1] = rgb[2] = -1;
    }
};
__forceinline__ __device__ Real cugetVal(cudaVars*vars, mode m, cuComplex &data, Real decay, bool islog){
  Real target = 0;
  switch(m){
    case IMAG: target = data.y*decay; break;
    case MOD: target = cuCabsf(data)*decay; break;
    case MOD2: target = (data.x*data.x+data.y*data.y)*decay; break;
    case PHASE: target = atan2(data.y,data.x)/2/M_PI+0.5; break;
    case PHASERAD: target = atan2(data.y,data.x); break;
    case REAL:{
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
    default: ;
  }
  if(target!=0){
    if(islog) target = log2f(target)/log2f(vars->rcolor)+1;
    target*=vars->rcolor;
  }
  return target;
}
__forceinline__ __device__ Real cugetVal(cudaVars* vars, mode m, Real &data, Real decay, bool islog){
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
  else if(m==MOD) ret = fabs(data)*decay;
  if(ret!=0){
    if(islog) ret = log2f(ret)/log2f(vars->rcolor)+1;
    ret*=vars->rcolor;
  }
  return ret;
}

cuFuncTemplate(process,(void* cudaData, pixeltype* cache, mode m, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0),(cudaData, cache, m, isFrequency, decay, islog, isFlip),{
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
  Real target = cugetVal(vars,m, ((T*)cudaData)[index],decay,islog);
  if(target < 0) target = 0;
  else if(target>=vars->rcolor) {
    target=vars->rcolor-1;
  }
  cache[targetx*cuda_column+targety] = floor(target);
})
template void process<Real>(void* cudaData, pixeltype* cache, mode m, bool isFrequency, Real decay, bool islog, bool isFlip);
template<> void process<complexFormat>(void* cudaData, pixeltype* cache, mode m, bool isFrequency, Real decay, bool islog, bool isFlip){
  processWrap<cuComplex><<<numBlocks, threadsPerBlock>>>(addVar(cudaData, cache, m, isFrequency, decay, islog, isFlip));
}

cuFunc(process_rgb,(void* cudaData, col_rgb* cache, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0),(cudaData, cache, isFrequency, decay, islog, isFlip),{
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
  cuComplex data = ((cuComplex*)cudaData)[index];
  Real mod = cuCabsf(data)*decay;
  char* col = (char*)(&(cache[targetx*cuda_column+targety]));
  if(mod > 1) mod = 1;
  Real phase = atan2(data.y,data.x)/2/M_PI+0.5; //0-1
  if(phase < 0) phase += 1;
  if(phase == 1) phase = 0;
  Real value = phase;
  value = 3*value + 0.5;
  value = fabs(value - round(value))/2 + 0.5;
  value = (1-mod) + mod*value;
  hsvToRGB(phase, mod, value, col);
})
