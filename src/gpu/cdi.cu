#include "cudaDefs.h"

cuFuncc(applyESWSupport,(complexFormat* ESW, complexFormat* ISW, complexFormat* ESWP, Real* length),(cuComplex* ESW, cuComplex* ISW, cuComplex* ESWP, Real* length),((cuComplex*)ESW,(cuComplex*)ISW,(cuComplex*)ESWP,length),{
  cuda1Idx()
    auto tmp = ISW[index];
  auto tmp2 = ESWP[index];
  auto sum = cuCaddf(tmp,ESWP[index]);
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
     if(x<vars->rows/3||x>vars->rows*2/3||y<cuda_column||y>2*cuda_column/3) factor = 0;
     ESW[index].x = factor*sum.x-tmp.x;
     ESW[index].y = factor*sum.y-tmp.y;

     ESW[index].x -= vars->beta_HIO*(1-factor)*sum.x;
     ESW[index].y -= vars->beta_HIO*(1-factor)*sum.y;
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
cuFuncc(applyESWMod,(complexFormat* ESW, Real* mod, complexFormat* amp, int noiseLevel),(cuComplex* ESW, Real* mod, cuComplex* amp, int noiseLevel),((cuComplex*)ESW,mod,(cuComplex*)amp,noiseLevel),{
  cuda1Idx()
    Real tolerance = 0;//1./vars->rcolor*vars->scale+1.5*sqrtf(noiseLevel)/vars->rcolor; // fluctuation caused by bit depth and noise
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
    if(mod2+tolerance < mod2s) factor = sqrt((mod2+tolerance)/mod2s);
    else if(mod2-tolerance > mod2s) factor = sqrt((mod2-tolerance)/mod2s);
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
  Real maximum = vars->scale*0.99;
  Real sourcedata = source[index].x;
  Real tolerance = 0.5/vars->rcolor*vars->scale;
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

