#include "holo.h"
#include "imgio.h"
#include "cudaDefs.h"
#include <curand_kernel.h>
#include "cuPlotter.h"
#include "cub_wrap.h"
#include "cudaConfig.h"

cuFuncc(applySupportBarHalf,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
  cudaIdx();
  int hr = cuda_row>>1;
  int hc = cuda_column>>1;
  if(x > hr) x -= hr;
  else x += hr;
  if(y > hc) y -= hc;
  else y += hc;
  if(spt[index] > vars->threshold || x + y > cuda_row)
    img[index].x = img[index].y = 0;
})


cuFuncc(applySupportBar_Flip,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
  cuda1Idx();
  if(spt[index] > vars->threshold){
    img[index].x *= -0.3;
    img[index].y *= -0.3;
  }
})

cuFuncc(applySupport,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
  cuda1Idx();
  if(spt[index] < vars->threshold)
    img[index].x = img[index].y = 0;
})

cuFuncc(dillate, (complexFormat* data, Real* support, int wid, int hit), (cuComplex* data, Real* support, int wid, int hit), ((cuComplex*)data,support,wid,hit),{
  cudaIdx();
  if(abs(data[index].x) < 0.5 && abs(data[index].y) < 0.5) return;
  int idxp = 0;
  for(int xp = 0; xp < cuda_row; xp++)
  for(int yp = 0; yp < cuda_column; yp++)
  {
    if(abs(xp - x) <= wid && abs(yp-y) <= hit) support[idxp] = 0;
    if(abs(x - xp) > cuda_row/2 || abs(y-yp)>cuda_column/2) support[idxp] = 0;
    idxp++;
  }
})

cuFuncc(applyModCorr, (complexFormat* obj, complexFormat* refer, Real* xcorrelation),(cuComplex* obj ,cuComplex* refer, Real* xcorrelation),((cuComplex*)obj,(cuComplex*)refer,xcorrelation),{
  cuda1Idx();
  cuComplex objtmp = obj[index];
  cuComplex reftmp = refer[index];
  if(reftmp.x == 0 && reftmp.y == 0) return;
  Real fact = xcorrelation[index]/2 - reftmp.x*objtmp.x - reftmp.y*objtmp.y;
  fact /= reftmp.x*reftmp.x + reftmp.y*reftmp.y;
  obj[index].x = objtmp.x + fact*reftmp.x;
  obj[index].y = objtmp.y + fact*reftmp.y;
})

cuFuncc(devideStar, (complexFormat* obj, complexFormat* refer, complexFormat* xcorrelation),(cuComplex* obj ,cuComplex* refer, cuComplex* xcorrelation),((cuComplex*)obj,(cuComplex*)refer,(cuComplex*)xcorrelation),{
  cuda1Idx();
  cuComplex xctmp = xcorrelation[index];
  cuComplex reftmp = refer[index];
  Real fact = max(sqSum(reftmp.x,reftmp.y),1e-4);
  xctmp = cuCmulf(xctmp, reftmp);
  obj[index].x = xctmp.x / fact;
  obj[index].y = xctmp.y / fact;
})

holo::holo(const char* configfile) : CDI(configfile){}
void holo::allocateMem_holo(){
  size_t sz = row*column*sizeof(Real);
  patternData_holo = (Real*)memMngr.borrowCache(sz);
  patternData_obj = (Real*)memMngr.borrowCache(sz);
  xcorrelation = (Real*)memMngr.borrowCache(sz);
  support_holo = (Real*)memMngr.borrowCache(sz);
  xcorrelation_support = (Real*)memMngr.borrowCache(sz);
  patternWave_holo = memMngr.borrowCache(sz*2);
  patternWave_obj = memMngr.borrowCache(sz*2);
  objectWave_holo = memMngr.borrowCache(sz*2);
}
void holo::calcXCorrelation(bool doplot){
  add( patternData_holo, patternData, -1);
  extendToComplex( patternData_holo, (complexFormat*)patternWave_holo);
  add( patternData_holo, patternData, 1);
  myFFT((complexFormat*)patternWave_holo,(complexFormat*)patternWave_holo);
  applyNorm( (complexFormat*)patternWave_holo, 1./(row*column));
  applyMaskBar( (complexFormat*)patternWave_holo, xcorrelation_support, 0.5);
  if(doplot) plt.plotComplex(patternWave_holo, MOD, 1, row*exposurepupil, "xcorrelation", 1, 0, 1);
  myIFFT((complexFormat*)patternWave_holo,(complexFormat*)patternWave_holo);
  getReal( xcorrelation, (complexFormat*)patternWave_holo);
}
void holo::calcXCorrelationHalf(bool doplot){
  add( patternData_holo, patternData, -1);
  extendToComplex( patternData_holo, (complexFormat*)patternWave_holo);
  add( patternData_holo, patternData, 1);
  myFFT((complexFormat*)patternWave_holo,(complexFormat*)patternWave_holo);
  applyNorm( (complexFormat*)patternWave_holo, 1./(row*column));
  applySupportBarHalf( (complexFormat*)patternWave_holo, xcorrelation_support);
  if(doplot) plt.plotComplex(patternWave_holo, MOD, 1, row*exposurepupil, "xcorrelation", 1, 0, 1);
  myIFFT((complexFormat*)patternWave_holo,(complexFormat*)patternWave_holo);  //OR*
}
void holo::initXCorrelation(){
  add( patternData_holo, patternData, -1);
  //plt.plotFloat(patternData_holo, MOD, 1, 1, "patternDiff", 1, 0, 1);
  extendToComplex( patternData_holo, (complexFormat*)patternWave_holo);
  add( patternData_holo, patternData, 1);
  myFFT((complexFormat*)patternWave_holo,(complexFormat*)patternWave_holo);
  applyNorm( (complexFormat*)patternWave_holo, 1./(row*column));

  rect cir;
  cir.startx=row/2-objrow;
  cir.starty=column/2-objcol;
  cir.endx=row/2+objrow;
  cir.endy=column/2+objcol;
  decltype(cir) *cuda_spt;
  cuda_spt = (decltype(cir)*)memMngr.borrowCache(sizeof(cir));
  cudaMemcpy(cuda_spt, &cir, sizeof(cir), cudaMemcpyHostToDevice);
  createMask( xcorrelation_support, cuda_spt, 1);
  
  applyMaskBar( (complexFormat*)patternWave_holo, xcorrelation_support, 0.5);
  plt.plotComplex(patternWave_holo, MOD, 1, row*exposurepupil, "xcorrelation_init", 1, 0, 1);
  myIFFT((complexFormat*)patternWave_holo,(complexFormat*)patternWave_holo);
  plt.plotComplex(patternWave_holo, REAL, 1, exposurepupil, "xcorrspt", 1, 0, 1);
  getReal( xcorrelation, (complexFormat*)patternWave_holo);

  cudaMemset(support_holo, 0, memMngr.getSize(support_holo));
  plt.plotComplex(objectWave, MOD2, 0, 1, "object_ref", 0, 0, 0);
  linearConst( support_holo, support_holo, 0, 1);
  dillate( (float _Complex*)objectWave, support_holo, row/oversampling-objrow-12, column/oversampling-objcol-18);
  plt.plotFloat(support_holo, MOD, 0, 1, "holospt");
}
void holo::simulate(){
  if(runSim) readObjectWave();
  else readPattern();
  init();
  allocateMem_holo();
  if(runSim){
    Real* d_intensity = 0;
    Real* d_phase = 0;
    readComplexWaveFront(pupil.Intensity, phaseModulation_pupil?pupil.Phase:0,d_intensity,d_phase,objrow,objcol);
    resize_cuda_image(row, column);
    Real phasefactor = M_PI*lambda*d/sq(pixelsize*row);
    printf("phase factor, %f, %f, %f, %d\n", lambda, d, pixelsize, row);
    createWaveFront( d_intensity, d_phase, (complexFormat*)objectWave_holo, objrow, objcol, (row/oversampling-objrow)/2, (column/oversampling-objcol)/2, phasefactor);
    add( (complexFormat*)objectWave_holo, (complexFormat*)objectWave, 1);
    propagate((complexFormat*)objectWave_holo, (complexFormat*)patternWave_holo, 1);
    getMod2( patternData_holo, (complexFormat*)patternWave_holo);
    plt.plotComplex(objectWave_holo, MOD2, 0, 1, "holoIntensity");
    plt.plotComplex(objectWave_holo, PHASE, 0, 1, "holoPhase");
    cudaMemset(objectWave_holo, 0, sizeof(complexFormat)*row*column);
    cudaMemset(patternWave_holo, 0, sizeof(complexFormat)*row*column);
    if(simCCDbit) applyPoissonNoise_WO( patternData_holo, noiseLevel, (curandStateMRG32k3a*)devstates, 1./exposurepupil);
    plt.plotFloat(patternData_holo, MOD, 1, exposurepupil, "holoPattern");
    plt.plotFloat(patternData_holo, MOD, 1, exposurepupil, "holoPatternlogsim",1);
  }else{
    objrow = objcol = 90;
    resize_cuda_image(row, column);
  }
  if(!runSim || simCCDbit) {
    Real* intensity = readImage(pupil.Pattern, row, column);
    cudaMemcpy(patternData_holo, intensity, memMngr.getSize(patternData_holo), cudaMemcpyHostToDevice);
    ccmemMngr.returnCache(intensity);
    applyNorm( patternData_holo, 1./exposurepupil);
    cudaConvertFO( patternData_holo);
  }
  if(doIteration){
    prepareIter();
    phaseRetrieve();
  }else if(runSim){
    propagate((complexFormat*)objectWave,(complexFormat*)patternWave, 1);
    getMod2(patternData, (complexFormat*)patternWave);
  }else{
    prepareIter();
  }
  plt.plotFloat(patternData_holo, MOD, 1, exposurepupil, "holoPatternlog",1, 0, 1);
  resize_cuda_image(row, column);
  init_cuda_image(rcolor, 1./exposurepupil);
  initXCorrelation();
  //iterate();
  noniterative();
};
void holo::iterate(){
  Real gaussianSigma = 3;
  int size = floor(gaussianSigma*6);
  size = ((size>>1)<<1)+1;
  const size_t sz = row*column*sizeof(Real);
  Real*  d_gaussianKernel = (Real*) memMngr.borrowCache(size*size*sizeof(Real));
  Real* cuda_objMod = (Real*) memMngr.borrowCache(sz);
  cudaMemset(patternWave_obj, 0, sz*2);
  AlgoParser algo(algorithm);
  int ialgo;
  for(int i = 0;; i++){
    ialgo = algo.next();
    if(ialgo < 0) break;
    if(ialgo == shrinkWrap){
      getMod2( cuda_objMod, (complexFormat*)objectWave_holo);
      applyGaussConv(cuda_objMod, support_holo, d_gaussianKernel, gaussianSigma);
      cudaVarLocal->threshold = findMax(support_holo,row*column)*shrinkThreshold;
      cudaMemcpy(cudaVar, cudaVarLocal, sizeof(cudaVars),cudaMemcpyHostToDevice);
      if(gaussianSigma>1.5) {
        gaussianSigma*=0.99;
      }
      continue;
    }
    if(ialgo == XCORRELATION){
      add( (complexFormat*)patternWave_holo, (complexFormat*)patternWave_obj, (complexFormat*)patternWave, 1);
      //applyMod( patternWave_holo, patternData_holo, useBS? beamstop:0, 1, nIter, 0);
      applyModAbs( (complexFormat*)patternWave_holo, patternData_holo);
      getMod2( patternData_holo, (complexFormat*)patternWave_holo);
      calcXCorrelation(0);
      applyModCorr( (complexFormat*)patternWave_obj, (complexFormat*)patternWave, xcorrelation);
      continue;
    }
    add( (complexFormat*)patternWave_holo, (complexFormat*)patternWave_obj, (complexFormat*)patternWave, 1.);
    //applyMod( patternWave_holo, patternData_holo, useBS? beamstop:0, 1, i, noiseLevel);
    applyModAbs( (complexFormat*)patternWave_holo, patternData_holo);
    add( (complexFormat*)patternWave_obj, (complexFormat*)patternWave_holo, (complexFormat*)patternWave, -1.);
    myIFFT((complexFormat*)patternWave_obj, (complexFormat*)patternWave_obj);
    applyNorm( (complexFormat*)patternWave_obj, 1./sqrt(row*column));
    applySupport(objectWave_holo, patternWave_obj, (Algorithm)ialgo, support_holo);
    myFFT((complexFormat*)objectWave_holo, (complexFormat*)patternWave_obj);
    applyNorm( (complexFormat*)patternWave_obj, 1./sqrt(row*column));
  }
  bitMap( support_holo, support_holo, cudaVarLocal->threshold);
  plt.plotFloat(support_holo, MOD, 0, 1, "support", 0);
  memMngr.returnCache(d_gaussianKernel);
  plt.plotComplex(objectWave_holo, MOD2, 0, 1, "object");
  plt.plotComplex(objectWave_holo, PHASE, 0, 1, "object_phase");
}
void holo::noniterative(){
  calcXCorrelationHalf(1);
  plt.plotComplex(patternWave_holo, MOD2, 1, 1, "xcorr_pat", 1, 0, 1);
  plt.plotComplex(patternWave, MOD2, 1, 1, "pat", 1, 0, 1);
  devideStar( (complexFormat*)patternWave_obj, (complexFormat*)patternWave, (complexFormat*)patternWave_holo);
  myIFFT((complexFormat*)patternWave_obj, (complexFormat*)objectWave_holo);
  plt.plotComplex(patternWave_obj, MOD2, 1, exposurepupil, "object_pattern", 1, 0, 1);
  applyNorm( (complexFormat*)objectWave_holo, 1./sqrt(row*column));
  //cudaConvertFO((complexFormat*)objectWave_holo);
  plt.saveComplex(objectWave_holo, "object");
  plt.plotComplex(objectWave_holo, MOD2, 0, 1, "object");
  plt.plotComplex(objectWave_holo, PHASE, 0, 1, "object_phase");
}
