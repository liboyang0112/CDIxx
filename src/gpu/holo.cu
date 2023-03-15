#include "holo.h"
#include "common.h"
#include "cuPlotter.h"

cuFunc(applySupportBar,(cudaVars* vars, complexFormat* img, Real* spt),(vars,img,spt),{
  cudaIdx();
  if(spt[index] > vars->threshold)
    img[index].x = img[index].y = 0;
})

cuFunc(applySupport,(cudaVars* vars, complexFormat* img, Real* spt),(vars,img,spt),{
  cudaIdx();
  if(spt[index] < vars->threshold)
    img[index].x = img[index].y = 0;
})

cuFunc(dillate, (cudaVars* vars, complexFormat* data, Real* support, int wid, int hit), (vars,data,support,wid,hit),{
  cudaIdx();
  if(abs(data[index].y) < 0.5 && abs(data[index].y) < 0.5) return;
  int idxp = 0;
  for(int xp = 0; xp < cuda_row; xp++)
  for(int yp = 0; yp < cuda_column; yp++)
  {
    if(abs(xp - x) < wid && abs(yp-y) < hit) support[idxp] = 1;
    if(abs(xp - x) > cuda_row/2 || abs(yp-y)>cuda_column/2) support[idxp] = 1;
    idxp++;
  }
})
cuFunc(applyModCorr, (cudaVars* vars, complexFormat* obj, complexFormat* refer, Real* xcorrelation),(vars,obj,refer,xcorrelation),{
  cudaIdx();
  complexFormat objtmp = obj[index];
  complexFormat reftmp = refer[index];
  Real maximum = vars->scale*0.99;
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
void holo::calcXCorrelation(){
  cudaF(add, patternData_holo, patternData, -1);
  cudaF(extendToComplex, patternData_holo, patternWave_holo);
  cudaF(add, patternData_holo, patternData, 1);
  myCufftExec(*plan, patternWave_holo,patternWave_holo,CUFFT_FORWARD);
  cudaF(applyNorm, patternWave_holo, 1./(row*column));

  rect cir;
  cir.startx=row/2-objrow;
  cir.starty=column/2-objcol;
  cir.endx=row/2+objrow;
  cir.endy=column/2+objcol;
  decltype(cir) *cuda_spt;
  cuda_spt = (decltype(cir)*)memMngr.borrowCache(sizeof(cir));
  cudaMemcpy(cuda_spt, &cir, sizeof(cir), cudaMemcpyHostToDevice);
  cudaF(createMask, xcorrelation_support, cuda_spt, 1);
  
  cudaF(applySupportBar, patternWave_holo, xcorrelation_support);
  myCufftExec(*plan, patternWave_holo,patternWave_holo,CUFFT_INVERSE);
  plt.plotComplex(patternWave_holo, REAL, 1, exposurepupil, "xcorrspt", 1);
  cudaF(getReal, xcorrelation, patternWave_holo);

  cudaMemset(support_holo, 0, memMngr.getSize(support_holo));
  cudaF(dillate, (complexFormat*)objectWave, support_holo, row/oversampling-objrow, column/oversampling-objcol);
  plt.plotFloat(support_holo, MOD, 0, 1, "holospt");
}
void holo::simulate(){
  readObjectWave();
  init();
  prepareIter();
  allocateMem_holo();
  Real* intensity = readImage(pupil.Intensity.c_str(), objrow, objcol);
  size_t sz = objrow*objcol*sizeof(Real);
  Real* d_intensity = (Real*)memMngr.borrowCache(sz); //use the memory allocated;
  cudaMemcpy(d_intensity, intensity, sz, cudaMemcpyHostToDevice);
  ccmemMngr.returnCache(intensity);
  Real* d_phase = 0;
  if(phaseModulation_pupil) {
    int tmprow,tmpcol;
    Real* phase = readImage(pupil.Phase.c_str(), tmprow,tmpcol);
    d_phase = (Real*)memMngr.borrowCache(sz);
    size_t tmpsz = tmprow*tmpcol*sizeof(Real);

    if(tmpsz!=sz){
      Real* d_phasetmp = (Real*)memMngr.borrowCache(tmpsz);
      gpuErrchk(cudaMemcpy(d_phasetmp,phase,tmpsz,cudaMemcpyHostToDevice));
      init_cuda_image(objrow, objcol);
      if(tmpsz > sz){
        cudaF(crop,d_phasetmp, d_phase, tmprow, tmpcol);
      }else{
        cudaF(pad,d_phasetmp, d_phase, tmprow, tmpcol);
      }
      memMngr.returnCache(d_phasetmp);
    }
    else {
      gpuErrchk(cudaMemcpy(d_phase,phase,sz,cudaMemcpyHostToDevice));
    }
    ccmemMngr.returnCache(phase);
  }
  init_cuda_image(row, column);
  cudaF(createWaveFront, d_intensity, d_phase, objectWave_holo, objrow, objcol, (row/oversampling-objrow)/2, (column/oversampling-objcol)/2);
  cudaF(add, objectWave_holo, (complexFormat*)objectWave, 1);
  plt.plotComplex(objectWave_holo, MOD2, 0, 1, "input");
  plt.plotComplex(objectWave_holo, PHASE, 0, 1, "input_phase");
  if(doIteration) phaseRetrieve();
  else propagate(patternWave, objectWave, 0);
  propagate(objectWave_holo, patternWave_holo, 1);
  cudaF(getMod2, patternData_holo, patternWave_holo);
  plt.plotFloat(patternData_holo, MOD, 1, exposurepupil, "holoPattern");
  if(simCCDbit) {
    Real* intensity = readImage("holoPattern.png", row, column);
    cudaMemcpy(patternData_holo, intensity, memMngr.getSize(patternData_holo), cudaMemcpyHostToDevice);
    ccmemMngr.returnCache(intensity);
    cudaF(applyPoissonNoise_WO, patternData_holo, noiseLevel, devstates, 1);
    cudaF(applyNorm, patternData_holo, 1./exposurepupil);
    cudaF(cudaConvertFO, patternData_holo);
  }
  calcXCorrelation();
  iterate();
};
void holo::iterate(){
  cudaMemset(patternWave_obj, 0, memMngr.getSize(patternWave_obj));
  for(int iter = 0; iter < nIter; iter++){
    cudaF(applyModCorr, patternWave_obj, patternWave, xcorrelation);
    cudaF(add, patternWave_obj, patternWave, 1);
    cudaF(applyMod, patternWave_obj, patternData_holo, useBS? beamstop:0, 1, iter, noiseLevel);
    cudaF(add, patternWave_obj, patternWave, -1);
    myCufftExec(*plan, patternWave_obj, patternWave_obj, CUFFT_INVERSE);
    cudaF(applyNorm, patternWave_obj, 1./(row*column));
    cudaF(applySupportBar, patternWave_obj, support_holo);
    myCufftExec(*plan, patternWave_obj, patternWave_obj, CUFFT_FORWARD);
  }
  propagate(patternWave_obj,objectWave_holo,0);
  plt.plotComplex(objectWave_holo, MOD2, 0, 1, "object");
  plt.plotComplex(objectWave_holo, PHASE, 0, 1, "object_phase");
}
