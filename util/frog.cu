#include "cudaConfig.h" //cuda related
#include "cudaDefs.h"
#include "cuPlotter.h" //plt
#include <complex>
#include <curand_kernel.h>
#include "cub_wrap.h"
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#define ALPHA 0.5
#define BETA 1
#define DELTA 1e-3
#define GAMMA 0.5
using namespace std;

Real* simTrace(int ndelay, int nspect){
  myDMalloc(Real, traces, nspect*ndelay);
  return traces;
}

void saveWave(const char* fname, complexFormat* ccE, int n){
  std::ofstream file1(fname, std::ios::out);
  for(int i = 0; i < n; i++){
    auto dat = ((complex<float>*)ccE)[i];
    file1<<i << " " << dat.real() << " " << dat.imag() << " " << abs(dat) << " " << arg(dat)<<std::endl;
  }
  file1.close();
}

cuFuncc(dgenTrace, (Real* gate, Real* E, complexFormat* fulltrace, Real* delay),(Real* gate, Real* E, cuComplex* fulltrace, Real* delay), (gate, E, (cuComplex*)fulltrace,delay), {
  cudaIdx();
  int tidx = y-delay[x];
  if(tidx >= cuda_column || tidx < 0) {
    fulltrace[index].x = 0;
  }else
    fulltrace[index].x = gate[tidx] * E[y];
  fulltrace[index].y = 0;
})

cuFuncc(dgencTraceSingle, (complexFormat* gate, complexFormat* E, complexFormat* trace, Real* delay, int i),(cuComplex* gate, cuComplex* E, cuComplex* trace, Real* delay, int i), ((cuComplex*)gate, (cuComplex*)E, (cuComplex*)trace,delay, i), {
  cuda1Idx();
  int tidx = index-delay[i];
  if(tidx >= cuda_row || tidx < 0) {
    trace[index].x = 0;
    trace[index].y = 0;
  }else
    trace[index] = cuCmulf(gate[tidx],E[index]);
})

cuFuncc(dgencTrace, (complexFormat* gate, complexFormat* E, complexFormat* fulltrace, Real* delay),(cuComplex* gate, cuComplex* E, cuComplex* fulltrace, Real* delay), ((cuComplex*)gate, (cuComplex*)E, (cuComplex*)fulltrace,delay), {
  cudaIdx();
  int tidx = y-delay[x];
  if(tidx >= cuda_column || tidx < 0) {
    fulltrace[index].x = 0;
    fulltrace[index].y = 0;
  }else
    fulltrace[index] = cuCmulf(gate[tidx],E[y]);
})


cuFuncc(genEComplex,(complexFormat* E),(cuComplex* E),((cuComplex*)E),{
  cuda1Idx();
  int rindex = cuda_row-index-1;
  int bias = index-cuda_row/2;
  Real sigma = 50;
  Real midf = 0;
  Real chirp = 2e-4;
  Real CEP = M_PI;
  Real phase = 2*M_PI*midf*index + CEP + 2*M_PI*chirp*(index-500)*(index-500);
  Real envolope = (exp(-sq(bias-50)/(2*sq(sigma)))+exp(-sq(bias+50)/(2*sq(sigma))));
  E[rindex].x = envolope*cos(phase);
  E[rindex].y = envolope*sin(phase);
})

cuFunc(genE,(Real* E),(E),{
  cuda1Idx();
  int bias = index-cuda_row/2;
  Real sigma = 20;
  Real chirp = 1e-5;
  Real midfreq = 22;
  Real CEP = M_PI;
  E[index] = (exp(-sq(bias-100)/(2*sq(sigma)))+exp(-sq(bias+100)/(2*sq(sigma))))*cos(2*M_PI/midfreq*index + 2*M_PI/chirp*index*index + CEP);
})

cuFunc(setDelay,(Real* delay),(delay),{
  cuda1Idx();
  delay[index] = Real(index)-cuda_row/2;
})

cuFuncc(convertFOy, (complexFormat* data),(cuComplex* data), ((cuComplex*)data), {
  cuda1Idx();
  int y = index%cuda_column;
  if(y >= cuda_column/2) return;
  cuComplex tmp = data[index];
  data[index] = data[index+cuda_column/2];
  data[index+cuda_column/2] = tmp;
})
cuFuncc(updateGE, (complexFormat* E, complexFormat* gate, complexFormat* trace, Real* delays, int i, Real alpha),(cuComplex* E, cuComplex* gate, cuComplex* trace, Real* delays, int i, Real alpha), ((cuComplex*)E,(cuComplex*)gate,(cuComplex*)trace,delays,i, alpha), {
  cuda1Idx();
  int tidx = index - delays[i];
  if(tidx >= cuda_row || tidx < 0) return;
  if(index < 100 || index > cuda_row-100) {
    E[index].x = 0;
    E[index].y = 0;
    return;
  }
  cuComplex tmp1 = E[index];
  tmp1.y = -tmp1.y;
  Real alpha1 = 0.1*alpha/(sqSum(tmp1.x,tmp1.y)+1e-2);
  tmp1 = cuCmulf(tmp1,trace[i*cuda_row+index]);

  gate[tidx].x += alpha1*tmp1.x;
  gate[tidx].y += alpha1*tmp1.y;

  cuComplex tmp = gate[tidx];
  tmp.y = -tmp.y;
  alpha /= (sqSum(tmp.x,tmp.y)+3e-3);
  tmp = cuCmulf(tmp,trace[i*cuda_row+index]);

  E[index].x += alpha*tmp.x;
  E[index].y += alpha*tmp.y;

})
cuFuncc(updateE, (complexFormat* E, complexFormat* gate, complexFormat* trace, Real* delays, int i, Real alpha),(cuComplex* E, cuComplex* gate, cuComplex* trace, Real* delays, int i, Real alpha), ((cuComplex*)E,(cuComplex*)gate,(cuComplex*)trace,delays,i, alpha), {
  cuda1Idx();
  int tidx = index - delays[i];
  if(tidx >= cuda_row || tidx < 0) return;
  if(index < 100 || index > cuda_row-100) {
    E[index].x = 0;
    E[index].y = 0;
    return;
  }
  cuComplex tmp = gate[tidx];
  tmp.y = -tmp.y;
  alpha /= sqrt(sqSum(tmp.x,tmp.y)+1e-1);
  tmp = cuCmulf(tmp,trace[index]);
  E[index].x += alpha*tmp.x;
  E[index].y += alpha*tmp.y;
})

cuFuncc(initGate, (complexFormat* gate),(cuComplex* gate), ((cuComplex*)gate), {
  cuda1Idx();
  int window = 100;
  if(index > cuda_row/2-window && index < cuda_row/2+window){
    gate[index].x = 1;
  }else gate[index].x = 0;
  gate[index].y = 0;
})

cuFuncc(removeHighFreq, (complexFormat* data),(cuComplex* data), ((cuComplex*)data), {
  cuda1Idx();
  if(index >= cuda_row/4 && index < cuda_row/4*3){
    data[index].x = 0;
    data[index].y = 0;
  }
})

cuFuncc(shiftmid, (complexFormat* Eprime, complexFormat* E, int tx),(cuComplex* Eprime, cuComplex* E, int tx),((cuComplex*)Eprime, (cuComplex*)E, tx),{
  cuda1Idx();
  int tidx = index+tx;
  if(tidx < 0 || tidx >= cuda_row) {
    E[index].x = 0;
    E[index].y = 0;
  }
  else E[index] = Eprime[tidx];
})

cuFuncc(average, (complexFormat* Eprime, complexFormat* E, Real gamma),(cuComplex* Eprime, cuComplex* E, Real gamma),((cuComplex*)Eprime, (cuComplex*)E, gamma),{
  cuda1Idx();
  Eprime[index].x = E[index].x = (Eprime[index].x+E[index].x)/2;
  Eprime[index].y = E[index].y = (Eprime[index].y+E[index].y)/2;
})

cuFuncc(stepMove, (complexFormat* Eprime, complexFormat* E, Real gamma),(cuComplex* Eprime, cuComplex* E, Real gamma),((cuComplex*)Eprime, (cuComplex*)E, gamma),{
  cuda1Idx();
  Real diff = Eprime[index].x - E[index].x;
  if(diff > gamma) diff = gamma;
  else if(diff < -gamma) diff = -gamma;
  Eprime[index].x -= diff;
  diff = Eprime[index].y - E[index].y;
  if(diff > gamma) diff = gamma;
  else if(diff < -gamma) diff = -gamma;
  Eprime[index].y -= diff;
})


void solveE(complexFormat* E, Real* traceIntensity, Real* spectrum, complexFormat* trace, Real* delays, int singleplan){
  int ndelay = cuda_imgsz.x;
  int nspect = cuda_imgsz.y;
  myDMalloc(complexFormat, ccE, nspect);
  complexFormat* gate = (complexFormat*) memMngr.borrowSame(E);
  complexFormat* Eprime = (complexFormat*) memMngr.borrowSame(E);
  complexFormat* traceprime = (complexFormat*) memMngr.borrowSame(E);
  resize_cuda_image(nspect,1);
  initGate(gate);
  myMemcpyD2D(E, gate, nspect*sizeof(complexFormat));
  int niter = 100;
  void* devstates = newRand(nspect);
  Real step = 0.9;
  resize_cuda_image(nspect,1);
  initRand(devstates, time(NULL));
  vector<int> sf(ndelay);
  for(int i = 0; i < ndelay; i++) sf[i] = i;
  for(int i = 0; i < niter; i++){
    shuffle(sf.begin(),sf.end(), std::default_random_engine(time(NULL)));
    for(int j = 0; j < ndelay; j++){
      int sfd = sf[j];
      dgencTraceSingle(gate, E, trace, delays, sfd);
      myFFTM(singleplan, trace, traceprime);
      applyModAbs(traceprime, traceIntensity+sfd*nspect, devstates);
      applyNorm(traceprime, 1./sqrt(nspect));
      myIFFTM(singleplan, traceprime, traceprime);
      add(traceprime, trace, -1);
      updateE(E, gate, traceprime, delays, sfd, step);
      myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
    }
    if(spectrum){
      myFFTM(singleplan, E, Eprime);
      applyModAbs(Eprime, spectrum, devstates);
      applyNorm(Eprime, 1./sqrt(nspect));
      myIFFTM(singleplan, Eprime, E);
    }
    myFFTM(singleplan, E, Eprime);
    removeHighFreq(Eprime);
    if(i %20 == 0) {
      myIFFTM(singleplan, Eprime, Eprime);
      Real mid = complex<float>(findMiddle(Eprime,nspect)).real();
      shiftmid(Eprime, E, mid*nspect);
    }else{
      myIFFTM(singleplan, Eprime, E);
    }
    applyNorm(E, 1./nspect);
    myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
  }
  resize_cuda_image(ndelay,nspect);
  dgencTrace(gate, E, trace, delays);
  myFFT(trace, trace);
  applyNorm(trace, 1./sqrt(nspect));
  convertFOy(trace);
}

void genTrace(complexFormat* E, complexFormat* fulltrace, Real* delays){
  dgencTrace(E, E, fulltrace,delays);
  myFFT(fulltrace, fulltrace);
  applyNorm(fulltrace, 1./sqrt(cuda_imgsz.y));
}

void genTrace(Real* E, complexFormat* fulltrace, Real* delays){
  dgenTrace(E, E, fulltrace,delays);
  myFFT(fulltrace, fulltrace);
  applyNorm(fulltrace, 1./sqrt(cuda_imgsz.y));
}

int main(int argc, char** argv )
{
  init_cuda_image();  //always needed
  int ndelay = 200;
  myDMalloc(Real, delays, ndelay);
  int nspect = 1000;
  int nfulldelay = 1000;
  myCuDMalloc(Real, d_fulldelays, nfulldelay);
  myCuDMalloc(Real, d_delays, ndelay);
  myCuDMalloc(complexFormat, d_cE, nspect);
  myCuDMalloc(complexFormat, d_spect, nspect);
  myCuDMalloc(Real, d_spectrum, nspect);
  myCuDMalloc(complexFormat, d_fulltraces, nspect*nfulldelay);
  myCuDMalloc(Real, d_traceIntensity, nspect*ndelay);
  myCuDMalloc(complexFormat, d_traces, nspect*ndelay);
  //Generate E, simulate trace
  resize_cuda_image(nspect,1);
  //myCuDMalloc(Real, d_E, nspect);
  //genE(d_E);
  genEComplex(d_cE);
  myDMalloc(complexFormat, ccE, nspect);
  myMemcpyD2H(ccE, d_cE, sizeof(complexFormat)*nspect);
  saveWave("input.txt", ccE, nspect);
  int singleplan;
  createPlan1d(&singleplan, nspect);
  myFFTM(singleplan, d_cE, d_spect);
  applyNorm(d_spect, 1./sqrt(nspect));
  getMod2(d_spectrum, d_spect);
  cudaConvertFO(d_spect);
  convertFOPhase(d_spect);
  myMemcpyD2H(ccE, d_spect, sizeof(complexFormat)*nspect);
  saveWave("inputSpect.txt", ccE, nspect);
  resize_cuda_image(nfulldelay,1);
  setDelay(d_fulldelays);
  init_fft(nspect,1,nfulldelay);
  resize_cuda_image(nfulldelay,nspect);
  plt.init(nfulldelay,nspect);
  //genTrace(d_E, d_fulltraces, d_fulldelays);
  genTrace(d_cE, d_fulltraces, d_fulldelays);
  convertFOy(d_fulltraces);
  plt.plotComplex(d_fulltraces, MOD2, 0, 1, "trace_truth", 1, 0, 1);
  //select delay, reconstruct E;
  srand(time(NULL));
  for(int i = 0; i < ndelay; i++){
    delays[i] = i*nfulldelay/ndelay - nfulldelay/2;
  }
  myMemcpyH2D(d_delays, delays, ndelay*sizeof(Real));
  resize_cuda_image(ndelay,nspect);
  plt.init(ndelay,nspect);
  init_fft(nspect,1,ndelay);
  genTrace(d_cE, d_traces, d_delays);
  getMod2(d_traceIntensity, d_traces);
  convertFOy(d_traces);
  plt.plotComplex(d_traces, MOD2, 0, 1, "trace_sampled", 1, 0, 1);
  clearCuMem(d_cE,  nspect*sizeof(complexFormat));
  clearCuMem(d_traces,  nspect*ndelay*sizeof(complexFormat));
  solveE(d_cE, d_traceIntensity, 0, d_traces, d_delays, singleplan);
  //solveE(d_cE, d_traceIntensity, d_spectrum, d_traces, d_delays, singleplan);
  plt.plotComplex(d_traces, MOD2, 0, 1, "trace_recon", 1, 0, 1);
  myMemcpyD2H(ccE, d_cE, sizeof(complexFormat)*nspect);
  saveWave("output.txt", ccE, nspect);
  myFFTM(singleplan, d_cE, d_spect);
  resize_cuda_image(nspect,1);
  applyNorm(d_spect, 1./sqrt(nspect));
  cudaConvertFO(d_spect);
  convertFOPhase(d_spect);
  myMemcpyD2H(ccE, d_spect, sizeof(complexFormat)*nspect);
  saveWave("outputSpect.txt", ccE, nspect);
  init_fft(nspect,1,nfulldelay);
  resize_cuda_image(nfulldelay,nspect);
  plt.init(nfulldelay,nspect);
  genTrace(d_cE, d_fulltraces, d_fulldelays);
  convertFOy(d_fulltraces);
  plt.plotComplex(d_fulltraces, MOD2, 0, 1, "trace_recon_full", 1, 0, 1);
}

