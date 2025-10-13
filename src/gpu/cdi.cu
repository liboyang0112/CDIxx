#include "format.hpp"
enum mode {MOD2,MOD, REAL, IMAG, PHASE, PHASERAD};
#include "cudaDefs_h.cu"
#include "memManager.hpp"
#include <curand_kernel.h>
__constant__ cudaVars vars;
cudaVars* cudaVarLocal = 0;
void init_cuda_image(int rcolor, Real scale){
  const int sz = sizeof(cudaVars);
  if(!cudaVarLocal){
    cudaFree(0);
    myMalloc(cudaVars, cudaVarLocal, 1);
    cudaVarLocal->threshold = 0.5;
    cudaVarLocal->beta_HIO = 1;
    cudaVarLocal->scale = scale;
    cudaVarLocal->rcolor = rcolor;
  }else{
    if(rcolor!=0) cudaVarLocal->rcolor = rcolor;
    if(scale==scale) cudaVarLocal->scale = scale;
  }
  cudaMemcpyToSymbol(vars, cudaVarLocal, sz);
};
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
__forceinline__ __device__ Real cugetVal(mode m, cuComplex &data, Real decay, bool islog){
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
          Real logv = log2f(ab)/log2f(vars.rcolor)+1;
          if(logv < 0) target = 0;
          else target = target*logv/(2*ab);
        }
      }
      return (target+0.5)*vars.rcolor;
    }
    default: ;
  }
  if(target!=0){
    if(islog) target = log2f(target)/log2f(vars.rcolor)+1;
    target*=vars.rcolor;
  }
  return target;
}
__forceinline__ __device__ Real cugetVal(mode m, Real &data, Real decay, bool islog){
  Real ret = 0;
  if(m==REAL) {
    ret = data*decay; //-1~1
    if(islog){
      if(ret!=0){
        Real ab = fabs(ret);
        Real logv = log2f(ab)/log2f(vars.rcolor)+1;
        if(logv < 0) ret = 0;
        else ret = ret*logv/(2*ab);
      }
    }
    return (ret+0.5)*vars.rcolor;
  }
  if(m==MOD2) ret = data*data*decay;
  else if(m==MOD) ret = fabs(data)*decay;
  if(ret!=0){
    if(islog) ret = log2f(ret)/log2f(vars.rcolor)+1;
    ret*=vars.rcolor;
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
  Real target = cugetVal(m, ((T*)cudaData)[index],decay,islog);
  if(target < 0) target = 0;
  else if(target>=vars.rcolor) {
    target=vars.rcolor-1;
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
  if(islog){
    if(mod!=0){
      Real ab = fabs(mod);
      Real logv = log2f(ab)/log2f(vars.rcolor)+1;
      if(logv < 0) mod = 0;
      else mod = mod*logv/(2*ab);
    }
  }
  Real phase = atan2(data.y,data.x)/2/M_PI+0.5; //0-1
  if(phase < 0) phase += 1;
  if(phase == 1) phase = 0;
  Real value = phase;
  value = 3*value + 0.5;
  value = fabs(value - round(value))/2 + 0.5;
  value = (1-mod) + mod*value;
  hsvToRGB(phase, mod, value, col);
})
void setThreshold(Real val){
  cudaMemcpyToSymbol(vars, &val, sizeof(Real), offsetof(cudaVars, threshold));
}
cuFunc(ccdRecord, (uint16_t* data, Real* wave, int noiseLevel, void* state, Real exposure),
    (data,wave,noiseLevel,state,exposure),{
    cuda1Idx()
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + vars.rcolor*wave[index]*exposure);
    if(dataint >= vars.rcolor) dataint = vars.rcolor-1;
    data[index] = dataint-noiseLevel;
    });
cuFuncc(ccdRecord, (uint16_t* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(uint16_t* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    (data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + vars.rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure);
    if(dataint >= vars.rcolor) dataint = vars.rcolor-1;
    data[index] = dataint-noiseLevel;
    });
cuFunc(ccdRecord, (Real* data, Real* wave, int noiseLevel, void* state, Real exposure, int rcolor),
    (data,wave,noiseLevel,state,exposure, rcolor),{
    cuda1Idx()
    if(rcolor == 0) rcolor = vars.rcolor;
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + rcolor*wave[index]*exposure);
    if(dataint >= rcolor) dataint = rcolor-1;
    data[index] = Real(dataint-noiseLevel)/rcolor;
    });
cuFuncc(ccdRecord, (Real* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(Real* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    (data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + vars.rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure);
    if(dataint >= vars.rcolor) dataint = vars.rcolor-1;
    data[index] = Real(dataint-noiseLevel)/vars.rcolor;
    });
cuFuncc(ccdRecord, (complexFormat* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(cuComplex* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    ((cuComplex*)data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + vars.rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure);
    if(dataint >= vars.rcolor) dataint = vars.rcolor-1;
    data[index].x = Real(dataint-noiseLevel)/vars.rcolor;
    data[index].y = 0;
    });
cuFunc(applyPoissonNoise,(Real* wave, Real noiseLevel, void* state, Real scale),
    (wave,noiseLevel,state,scale),{
    cuda1Idx()
    if(scale==0) scale = vars.scale;
    wave[index]+=scale*(curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel)-noiseLevel)/vars.rcolor;
    })
cuFunc(applyPoissonNoise_WO,(Real* wave, Real noiseLevel, void* state, Real scale),
    (wave,noiseLevel,state,scale),{
    cuda1Idx()
    if(scale==0) scale = vars.scale;
    wave[index]=scale*(int(wave[index]*vars.rcolor/scale) + curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel)-noiseLevel)/vars.rcolor;
    })

cuFuncc(bitMap,(Real* store, complexFormat* amp, Real threshold),(Real* store, cuComplex* amp, Real threshold),(store,(cuComplex*)amp, threshold),{
    cuda1Idx()
    if(threshold == 0) threshold = vars.threshold;
    cuComplex tmp = amp[index];
    store[index] = tmp.x*tmp.x+tmp.y*tmp.y > threshold*threshold;
    })

cuFunc(bitMap,(Real* store, Real* amp, Real threshold),(store,amp, threshold),{
    cuda1Idx()
    if(threshold == 0) threshold = vars.threshold;
    store[index] = amp[index] > threshold;
    })

cuFuncc(applyMod,
    (complexFormat* source, Real* target, Real *bs, int noiseLevel, Real bsnorm),
    (cuComplex* source, Real* target, Real *bs, int noiseLevel, Real bsnorm),
    ((cuComplex*)source, target, bs, noiseLevel, bsnorm),
{
    cuda1Idx();

    // Load data early
    cuComplex sourcedata = source[index];
    Real mod2 = fmaxf(0.0f, target[index]);

    // Early return if blocked by bs flag (common case?)
    if (bs && bs[index] > 0.5f) {
        // Only write back scaled value if bsnorm was applied
        if (bsnorm != 1.0f) {
            sourcedata.x *= bsnorm;
            sourcedata.y *= bsnorm;
            source[index] = sourcedata;
        }
        return;
    }
    Real rx = sourcedata.x;
    Real ry = sourcedata.y;
    if (bsnorm != 1.0f) {
        rx *= bsnorm;
        ry *= bsnorm;
    }
    Real srcmod2 = rx*rx + ry*ry;
    Real maximum = vars.scale * 0.95f;
    if (mod2 >= maximum) {
        mod2 = fmaxf(maximum, srcmod2);
    }
    Real tolerance = (1.0f + sqrtf((Real)noiseLevel)) * vars.scale / vars.rcolor;
    Real diff = mod2 - srcmod2;
    Real val = mod2;
    if (diff > tolerance) {
        val -= tolerance;
    } else if (diff < -tolerance) {
        val += tolerance;
    }
    if (srcmod2 == 0.0f) {
        source[index] = make_cuComplex(val, 0.0f);
        return;
    }
    Real scale = sqrtf(fmaxf(val, 0.0f) / srcmod2); // Clamp val to avoid NaN
    source[index] = make_cuComplex(scale * rx, scale * ry);
})
cuFuncc(applyRandomPhase,(complexFormat* wave, Real* beamstop, void* state),(cuComplex* wave, Real* beamstop, void* state),
    ((cuComplex*)wave, beamstop, state),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    if(beamstop && beamstop[index]>vars.threshold) {
    tmp.x = tmp.y = 0;
    }
    else{
    Real mod = cuCabsf(wave[index]);
    Real randphase = (curand_uniform((curandStateMRG32k3a*)state+index)>0.5)*M_PI;
    Real c,s;
    sincosf(randphase, &s, &c);
    tmp.x = mod*c;
    tmp.y = mod*s;
    }
    wave[index] = tmp;
    })
cuFuncc(takeMod2Diff,(complexFormat* a, Real* b, Real *output, Real *bs),(cuComplex* a, Real* b, Real *output, Real *bs),((cuComplex*)a,b,output,bs),{
    cuda1Idx()
    Real mod2 = sq(a[index].x)+sq(a[index].y);
    Real tmp = b[index]-mod2;
    if(bs&&bs[index]>0.5) tmp=0;
    else if(b[index]>vars.scale) tmp = vars.scale-mod2;
    output[index] = tmp;
    })

cuFuncc(takeMod2Sum,(complexFormat* a, Real* b),(cuComplex* a, Real* b),((cuComplex*)a,b),{
    cuda1Idx()
    Real tmp = b[index]+sq(a[index].x)+sq(a[index].y);
    if(tmp<0) tmp=0;
    b[index] = tmp;
    })
cuFunc(ApplyHIOSupport,
    (void *gkp1, void *gkprime, Real *spt),
    (gkp1,gkprime,spt),
{
  cuda1Idx();
  cuComplex &rhonp1 = ((cuComplex*)gkp1)[index];
  cuComplex &rhoprime = ((cuComplex*)gkprime)[index];
  bool insideS = spt[index] > vars.threshold;
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= vars.beta_HIO*rhoprime.x;
    rhonp1.y -= vars.beta_HIO*rhoprime.y;
  }
})
cuFunc(ApplyFHIOSupport,
    (void *gkp1, void *gkprime, Real *spt),
    (gkp1,gkprime,spt),
{
  cuda1Idx();
  cuComplex &rhonp1 = ((cuComplex*)gkp1)[index];
  cuComplex &rhoprime = ((cuComplex*)gkprime)[index];
  bool insideS = spt[index] > vars.threshold;
  if(insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    rhonp1.y += 1.9*(rhoprime.y-rhonp1.y);
  }else{
    rhonp1.x -= 1.2*rhoprime.x;
    rhonp1.y -= 1.2*rhoprime.y;
  }
})
cuFunc(ApplyRAARSupport,
    (void *gkp1, void *gkprime, Real *spt),
    (gkp1,gkprime,spt),
{
  cuda1Idx();
  cuComplex &rhonp1 = ((cuComplex*)gkp1)[index];
  cuComplex &rhoprime = ((cuComplex*)gkprime)[index];
  bool insideS = spt[index] > vars.threshold;
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{

    rhonp1.x = vars.beta_HIO*rhonp1.x+(1-2*vars.beta_HIO)*rhoprime.x;
    rhonp1.y = vars.beta_HIO*rhonp1.y+(1-2*vars.beta_HIO)*rhoprime.y;
    //    rhonp1.x = vars.beta_HIO*(rhonp1.x-rhoprime.x);
    //    rhonp1.y = vars.beta_HIO*(rhonp1.y-rhoprime.y);
  }
})
cuFunc(ApplyPOSERSupport,
    (void *gkp1, void *gkprime, Real *spt),
    (gkp1,gkprime,spt),
{
  cuda1Idx();
  cuComplex &rhonp1 = ((cuComplex*)gkp1)[index];
  cuComplex &rhoprime = ((cuComplex*)gkprime)[index];
  bool insideS = spt[index] > vars.threshold;
  if(insideS && rhoprime.x > 0){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x = 0;
    rhonp1.y = 0;
  }
})
cuFunc(ApplyERSupport,
    (void *gkp1, void *gkprime, Real *spt),
    (gkp1,gkprime,spt),
{
  cuda1Idx();
  cuComplex &rhonp1 = ((cuComplex*)gkp1)[index];
  cuComplex &rhoprime = ((cuComplex*)gkprime)[index];
  bool insideS = spt[index] > vars.threshold;
  if(insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    rhonp1.y += 1.9*(rhoprime.y-rhonp1.y);
  }else{
    rhonp1.x = 0;
    rhonp1.y = 0;
  }
})
cuFunc(ApplyPOSHIOSupport,
    (void *gkp1, void *gkprime, Real *spt),
    (gkp1,gkprime,spt),
{
  cuda1Idx();
  cuComplex &rhonp1 = ((cuComplex*)gkp1)[index];
  cuComplex &rhoprime = ((cuComplex*)gkprime)[index];
  bool insideS = spt[index] > vars.threshold;
  if(rhoprime.x > 0 && insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    //rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= vars.beta_HIO*rhoprime.x;
  }
  rhonp1.y -= vars.beta_HIO*rhoprime.y;
})
cuFunc(ApplyPOS0HIOSupport,
    (void *gkp1, void *gkprime, Real *spt),
    (gkp1,gkprime,spt),
{
  cuda1Idx();
  cuComplex &rhonp1 = ((cuComplex*)gkp1)[index];
  cuComplex &rhoprime = ((cuComplex*)gkprime)[index];
  bool insideS = spt[index] > vars.threshold;
  if(rhoprime.x > 0 && insideS){
    rhonp1.x = rhoprime.x;
    //rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= vars.beta_HIO*rhoprime.x;
  }
  rhonp1.y -= vars.beta_HIO*rhoprime.y;
})
cuFuncc(overExposureZeroGrad, (complexFormat* deltab, complexFormat* b),(cuComplex* deltab, cuComplex* b),((cuComplex*)deltab, (cuComplex*)b),{
    cuda1Idx();
    if(b[index].x >= vars.scale*0.99 && deltab[index].x < 0) deltab[index].x = 0;
    deltab[index].y = 0;
    })

cuFuncc(applySupportBarHalf,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
    cudaIdx();
    int hr = cuda_row>>1;
    int hc = cuda_column>>1;
    if(x > hr) x -= hr;
    else x += hr;
    if(y > hc) y -= hc;
    else y += hc;
    if(spt[index] > vars.threshold || x + y > cuda_row)
    img[index].x = img[index].y = 0;
    })


cuFuncc(applySupportBar_Flip,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
    cuda1Idx();
    if(spt[index] > vars.threshold){
    img[index].x *= -0.3;
    img[index].y *= -0.3;
    }
    })

cuFunc(applySupport,(Real* image, Real* support),(image,support),{
  cuda1Idx();
  if(support[index] > vars.threshold) image[index] = 0;
})
cuFuncc(applySupport,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
    cuda1Idx();
    if(spt[index] < vars.threshold)
    img[index].x = img[index].y = 0;
    })
cuFuncc(applyESWSupport,(complexFormat* ESW, complexFormat* ISW, complexFormat* ESWP),(cuComplex* ESW, cuComplex* ISW, cuComplex* ESWP),((cuComplex*)ESW,(cuComplex*)ISW,(cuComplex*)ESWP),{
  cuda1Idx()
    auto tmp = ISW[index];
  auto tmp2 = ESWP[index];
  //auto sum = cuCaddf(tmp,ESWP[index]);
  //these are for amplitude modulation only
  Real prod = tmp.x*tmp2.x+tmp.y*tmp2.y;
  if(prod>0) prod=0;
  if(prod<-2) prod = -2;
  auto rmod2 = 1./(tmp.x*tmp.x+tmp.y*tmp.y);
  ESW[index].x = prod*tmp.x*rmod2;
  ESW[index].y = prod*tmp.y*rmod2;
  /*
     if(cuCabsf(tmp) > cuCabsf(sum)) {
     ESW[index] = ESWP[index];
     length[index] = 0;
     return;
     }
     Real factor = cuCabsf(tmp)/cuCabsf(sum);
     if(x<vars.rows/3||x>vars.rows*2/3||y<cuda_column||y>2*cuda_column/3) factor = 0;
     ESW[index].x = factor*sum.x-tmp.x;
     ESW[index].y = factor*sum.y-tmp.y;

     ESW[index].x -= vars.beta_HIO*(1-factor)*sum.x;
     ESW[index].y -= vars.beta_HIO*(1-factor)*sum.y;
     length[index] = 1;
   */
})
cuFuncc(initESW,(complexFormat* ESW, Real* mod, complexFormat* amp),(cuComplex* ESW, Real* mod, cuComplex* amp),((cuComplex*)ESW,mod,(cuComplex*)amp),{
  cuda1Idx()
    auto tmp = amp[index];
  if(cuCabsf(tmp)<=1e-10) {
    ESW[index] = tmp;
    return;
  }
  if(mod[index]<=0) {
    ESW[index].x = -tmp.x;
    ESW[index].y = -tmp.y;
    return;
  }
  Real factor = sqrtf(mod[index])/cuCabsf(tmp)-1;
  ESW[index].x = factor*tmp.x;
  ESW[index].y = factor*tmp.y;
})
cuFuncc(applyESWMod,(complexFormat* ESW, Real* mod, complexFormat* amp),(cuComplex* ESW, Real* mod, cuComplex* amp),((cuComplex*)ESW,mod,(cuComplex*)amp),{
  cuda1Idx()
    Real tolerance = 0;//1./vars.rcolor*vars.scale+1.5*sqrtf(noiseLevel)/vars.rcolor; // fluctuation caused by bit depth and noise
  auto tmp = amp[index];
  auto sum = cuCaddf(ESW[index],tmp);
  Real mod2 = mod[index];
  if(mod2<=0){
    ESW[index].x = -tmp.x;
    ESW[index].y = -tmp.y;
    return;
  }
  Real factor = 0;
  if(cuCabsf(sum)>1e-10){
    //factor = mod[index]/cuCabsf(sum);
    Real mod2s = sum.x*sum.x+sum.y*sum.y;
    if(mod2+tolerance < mod2s) factor = sqrtf((mod2+tolerance)/mod2s);
    else if(mod2-tolerance > mod2s) factor = sqrtf((mod2-tolerance)/mod2s);
    else factor=1;
  }
  //if(mod[index] >= 0.99) factor = max(0.99/cuCabsf(sum), 1.);
  //printf("factor=%f, mod=%f, sum=%f\n", factor, mod[index], cuCabsf(sum));
  ESW[index].x = factor*sum.x-tmp.x;
  ESW[index].y = factor*sum.y-tmp.y;
})

cuFuncc(calcESW,(complexFormat* sample, complexFormat* ISW),(cuComplex* sample, cuComplex* ISW), ((cuComplex*)sample,(cuComplex*)ISW),{
  cuda1Idx()
    cuComplex tmp = sample[index];
  tmp.x = -tmp.x;  // Here we reverse the image, use tmp.x = tmp.x - 1 otherwise;
                   //Real ttmp = tmp.y;
                   //tmp.y=tmp.x;   // We are ignoring the factor (-i) each time we do fresnel propagation, which causes this transform in the ISW. ISW=iA ->  ESW=(O-1)A=(i-iO)ISW
                   //tmp.x=ttmp;
  sample[index]=cuCmulf(tmp,ISW[index]);
})

cuFuncc(calcO,(complexFormat* ESW, complexFormat* ISW),(cuComplex* ESW, cuComplex* ISW), ((cuComplex*)ESW,(cuComplex*)ISW),{
  cuda1Idx()
    if(cuCabsf(ISW[index])<1e-4) {
      ESW[index].x = 0;
      ESW[index].y = 0;
      return;
    }
  cuComplex tmp = cuCdivf(ESW[index],ISW[index]);
  /*
     Real ttmp = tmp.y;
     tmp.y=tmp.x;
     tmp.x=1-ttmp;
   */
  ESW[index].x=1+tmp.x;
  ESW[index].y=tmp.y;
})

cuFuncc(applyAutoCorrelationMod,(complexFormat* source,complexFormat* target, Real *bs = 0),(cuComplex* source, cuComplex* target, Real *bs),((cuComplex*)source,(cuComplex*)target,bs),{
  cuda1Idx()
  Real targetdata = target[index].x;
  Real retval = targetdata;
  source[index].y = 0;
  Real maximum = vars.scale*0.99;
  Real sourcedata = source[index].x;
  Real tolerance = 0.5/vars.rcolor*vars.scale;
  Real diff = sourcedata-targetdata;
  if(bs && bs[index]>0.5) {
    if(targetdata<0) target[index].x = 0;
    return;
  }
  if(diff>tolerance){
    retval = targetdata+tolerance;
  }else if(diff < -tolerance ){
    retval = targetdata-tolerance;
  }else{
    retval = targetdata;
  }
  if(targetdata>=maximum) {
    retval = max(sourcedata,maximum);
  }
  source[index].x = retval;
})

