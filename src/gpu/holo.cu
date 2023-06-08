#include "holo.h"
#include "common.h"
#include "cuPlotter.h"
#include "cub_wrap.h"

cuFunc(applySupportBar,(complexFormat* img, Real* spt),(img,spt),{
  cudaIdx();
  if(spt[index] > vars->threshold)
    img[index].x = img[index].y = 0;
})

cuFunc(applySupportBar_Flip,(complexFormat* img, Real* spt),(img,spt),{
  cudaIdx();
  if(spt[index] > vars->threshold){
    img[index].x *= -0.3;
    img[index].y *= -0.3;
  }
})

cuFunc(applySupport,(complexFormat* img, Real* spt),(img,spt),{
  cudaIdx();
  if(spt[index] < vars->threshold)
    img[index].x = img[index].y = 0;
})

cuFunc(dillate, (complexFormat* data, Real* support, int wid, int hit), (data,support,wid,hit),{
  cudaIdx();
  if(abs(data[index].y) < 0.5 && abs(data[index].y) < 0.5) return;
  int idxp = 0;
  for(int xp = 0; xp < cuda_row; xp++)
  for(int yp = 0; yp < cuda_column; yp++)
  {
    if(abs(xp - x) <= wid && abs(yp-y) <= hit) support[idxp] = 1;
    if(abs(x - xp) > cuda_row/2 || abs(y-yp)>cuda_column/2) support[idxp] = 1;
    idxp++;
  }
})
cuFunc(applyModCorr, (complexFormat* obj, complexFormat* refer, Real* xcorrelation),(obj,refer,xcorrelation),{
  cudaIdx();
  complexFormat objtmp = obj[index];
  complexFormat reftmp = refer[index];
  Real fact = xcorrelation[index]/2 - reftmp.x*objtmp.x - reftmp.y*objtmp.y;
  fact /= reftmp.x*reftmp.x + reftmp.y*reftmp.y;
  obj[index].x = objtmp.x + fact*reftmp.x;
  obj[index].y = objtmp.y + fact*reftmp.y;
})

holo::holo(const char* configfile) : CDI(configfile){}
void holo::allocateMem_holo(){
  size_t sz = row*column*sizeof(Real);
  patternData_holo = (Real*)memMngr.borrowCache(sz);
  patternData_obj = (Real*)memMngr.borrowCache(sz);
  xcorrelation = (Real*)memMngr.borrowCache(sz);
  support_holo = (Real*)memMngr.borrowCache(sz);
  xcorrelation_support = (Real*)memMngr.borrowCache(sz);
  patternWave_holo = (complexFormat*)memMngr.borrowCache(sz*2);
  patternWave_obj = (complexFormat*)memMngr.borrowCache(sz*2);
  objectWave_holo = (complexFormat*)memMngr.borrowCache(sz*2);
}
void holo::calcXCorrelation(bool doplot){
  add( patternData_holo, patternData, -1);
  extendToComplex( patternData_holo, patternWave_holo);
  add( patternData_holo, patternData, 1);
  myCufftExec(*plan, patternWave_holo,patternWave_holo,CUFFT_FORWARD);
  applyNorm( patternWave_holo, 1./(row*column));
  applySupportBar( patternWave_holo, xcorrelation_support);
  if(doplot) plt.plotComplex(patternWave_holo, MOD, 1, row*exposurepupil, "xcorrelation", 1);
  myCufftExec(*plan, patternWave_holo,patternWave_holo,CUFFT_INVERSE);
  getReal( xcorrelation, patternWave_holo);

}
void holo::initXCorrelation(){
  add( patternData_holo, patternData, -1);
  extendToComplex( patternData_holo, patternWave_holo);
  add( patternData_holo, patternData, 1);
  myCufftExec(*plan, patternWave_holo,patternWave_holo,CUFFT_FORWARD);
  applyNorm( patternWave_holo, 1./(row*column));

  rect cir;
  cir.startx=row/2-objrow;
  cir.starty=column/2-objcol;
  cir.endx=row/2+objrow;
  cir.endy=column/2+objcol;
  decltype(cir) *cuda_spt;
  cuda_spt = (decltype(cir)*)memMngr.borrowCache(sizeof(cir));
  cudaMemcpy(cuda_spt, &cir, sizeof(cir), cudaMemcpyHostToDevice);
  createMask( xcorrelation_support, cuda_spt, 1);
  
  applySupportBar( patternWave_holo, xcorrelation_support);
  plt.plotComplex(patternWave_holo, MOD, 1, row*exposurepupil, "xcorrelation_init", 1);
  myCufftExec(*plan, patternWave_holo,patternWave_holo,CUFFT_INVERSE);
  plt.plotComplex(patternWave_holo, REAL, 1, exposurepupil, "xcorrspt", 1);
  getReal( xcorrelation, patternWave_holo);

  cudaMemset(support_holo, 0, memMngr.getSize(support_holo));
  dillate( (complexFormat*)objectWave, support_holo, row/oversampling-objrow-12, column/oversampling-objcol-18);
  linearConst( support_holo, support_holo, -1, 1);
  plt.plotFloat(support_holo, MOD, 0, 1, "holospt");
}
void holo::simulate(){
  readObjectWave();
  init();
  allocateMem_holo();
  if(runSim){
    Real* d_intensity = 0;
    Real* d_phase = 0;
    readComplexWaveFront(pupil.Intensity, phaseModulation_pupil?pupil.Phase:0,d_intensity,d_phase,objrow,objcol);
    resize_cuda_image(row, column);
    createWaveFront( d_intensity, d_phase, objectWave_holo, objrow, objcol, (row/oversampling-objrow)/2, (column/oversampling-objcol)/2);
    add( objectWave_holo, (complexFormat*)objectWave, 1);
    propagate(objectWave_holo, patternWave_holo, 1);
    getMod2( patternData_holo, patternWave_holo);
    cudaMemset(objectWave_holo, 0, sizeof(complexFormat)*row*column);
    cudaMemset(patternWave_holo, 0, sizeof(complexFormat)*row*column);
    if(simCCDbit) applyPoissonNoise_WO( patternData_holo, noiseLevel, devstates, 1./exposurepupil);
    plt.plotFloat(patternData_holo, MOD, 1, exposurepupil, "holoPattern");
    plt.plotFloat(patternData_holo, MOD, 1, exposurepupil, "holoPatternlogsim",1);
  }else{
    readPattern();
    objrow = objcol = 150;
    resize_cuda_image(row, column);
  }
  if(!runSim || simCCDbit) {
    Real* intensity = readImage("holoPattern.png", row, column);
    cudaMemcpy(patternData_holo, intensity, memMngr.getSize(patternData_holo), cudaMemcpyHostToDevice);
    ccmemMngr.returnCache(intensity);
    applyNorm( patternData_holo, 1./exposurepupil);
    cudaConvertFO( patternData_holo);
  }
  prepareIter();
  if(doIteration) phaseRetrieve();
  else propagate(patternWave, objectWave, 0);
  plt.plotFloat(patternData_holo, MOD, 1, exposurepupil, "holoPatternlog",1);
  resize_cuda_image(row, column);
  init_cuda_image(rcolor, 1./exposurepupil);
  initXCorrelation();
  iterate();
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
      getMod2( cuda_objMod, objectWave_holo);
      applyGaussConv(cuda_objMod, support_holo, d_gaussianKernel, gaussianSigma);
      cudaVarLocal->threshold = findMax(support_holo,row*column)*shrinkThreshold;
      cudaMemcpy(cudaVar, cudaVarLocal, sizeof(cudaVars),cudaMemcpyHostToDevice);
      if(gaussianSigma>1.5) {
        gaussianSigma*=0.99;
      }
      continue;
    }
    if(ialgo == XCORRELATION){
      add( patternWave_holo, patternWave_obj, patternWave, 1);
      applyMod( patternWave_holo, patternData_holo, useBS? beamstop:0, 1, nIter, 0);
      getMod2( patternData_holo, patternWave_holo);
      calcXCorrelation(0);
      applyModCorr( patternWave_obj, (complexFormat*)patternWave, xcorrelation);
      continue;
    }
    add( patternWave_holo, patternWave_obj, patternWave, 1.);
    applyMod( patternWave_holo, patternData_holo, useBS? beamstop:0, 1, i, noiseLevel);
    add( patternWave_obj, patternWave_holo, patternWave, -1.);
    myCufftExec(*plan, patternWave_obj, patternWave_obj, CUFFT_INVERSE);
    applyNorm( patternWave_obj, 1./sqrt(row*column));
    applySupport( objectWave_holo, patternWave_obj, (Algorithm)ialgo, support_holo);
    myCufftExec(*plan, objectWave_holo, patternWave_obj, CUFFT_FORWARD);
    applyNorm( patternWave_obj, 1./sqrt(row*column));
  }
  bitMap( support_holo, support_holo, cudaVarLocal->threshold);
  plt.plotFloat(support_holo, MOD, 0, 1, "support", 0);
  memMngr.returnCache(d_gaussianKernel);
  plt.plotComplex(objectWave_holo, MOD2, 0, 1, "object");
  plt.plotComplex(objectWave_holo, PHASE, 0, 1, "object_phase");
}
