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

cuFuncc(dgencTraceSingle, (complexFormat* gate, complexFormat* E, complexFormat* trace, Real delay),(cuComplex* gate, cuComplex* E, cuComplex* trace, Real delay), ((cuComplex*)gate, (cuComplex*)E, (cuComplex*)trace,delay), {
  cuda1Idx();
  int tidx = index-delay;
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
  int bias = index-cuda_row/2;
  Real sigma = 5;
  Real midf = 0;
  Real chirp = 2e-3;
  Real CEP = M_PI;
  Real phase = 2*M_PI*midf*index + CEP + 2*M_PI*chirp*bias*bias;
  Real envolope = (exp(-sq(bias-10)/(2*sq(sigma)))+0.5*exp(-sq(bias+10)/(2*sq(sigma))));
  E[index].x = envolope*cos(phase);
  E[index].y = envolope*sin(phase);
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
  delay[index] = index-cuda_row/2;
})

cuFuncc(convertFOy, (complexFormat* data),(cuComplex* data), ((cuComplex*)data), {
  cuda1Idx();
  int y = index%cuda_column;
  if(y >= cuda_column/2) return;
  cuComplex tmp = data[index];
  data[index] = data[index+cuda_column/2];
  data[index+cuda_column/2] = tmp;
})
cuFuncc(applySoftThreshold, (complexFormat* data, Real thres),(cuComplex* data, Real thres), ((cuComplex*)data, thres), {
  cuda1Idx();
  Real mod = cuCabsf(data[index]);
  if(mod > thres){
    mod = (1-thres/cuCabsf(data[index]))/cuda_row;
    data[index].x *= mod;
    data[index].y *= mod;
  }else data[index].x = data[index].y = 0;
})
cuFuncc(updateGE, (complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha),(cuComplex* E, cuComplex* gate, cuComplex* trace, Real delay, Real alpha), ((cuComplex*)E,(cuComplex*)gate,(cuComplex*)trace,delay, alpha), {
  cuda1Idx();
  int tidx = index - delay;
  if(tidx >= cuda_row || tidx < 0) return;
  cuComplex tmp = E[index];
  tmp.y = -tmp.y;
  tmp = cuCmulf(tmp,trace[index]);
  gate[tidx].x += alpha*tmp.x;
  gate[tidx].y += alpha*tmp.y;

  tmp = gate[tidx];
  tmp.y = -tmp.y;
  tmp = cuCmulf(tmp,trace[index]);

  E[index].x += alpha*tmp.x;
  E[index].y += alpha*tmp.y;

})
cuFuncc(updateE, (complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha),(cuComplex* E, cuComplex* gate, cuComplex* trace, Real delay, Real alpha), ((cuComplex*)E,(cuComplex*)gate,(cuComplex*)trace,delay, alpha), {
  cuda1Idx();
  int tidx = index - delay;
  if(tidx >= cuda_row || tidx < 0) return;
  cuComplex tmp = gate[tidx];
  tmp.y = -tmp.y;
  tmp = cuCmulf(tmp,trace[index]);
  E[index].x += alpha*tmp.x;
  E[index].y += alpha*tmp.y;
})

cuFuncc(initE, (complexFormat* gate),(cuComplex* gate), ((cuComplex*)gate), {
  cuda1Idx();
  int window = 32;
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

void solveE(complexFormat* E, Real* traceIntensity, Real* spectrum, complexFormat* trace, Real* delays, int nfulldelay, Real* fulldelays, int singleplan){
  int ndelay = cuda_imgsz.x;
  int nspect = cuda_imgsz.y;
  myDMalloc(complexFormat, ccE, nspect);
  complexFormat* gate = trace+nspect;
  complexFormat* Eprime = gate+nspect;
  complexFormat* traceprime = Eprime+nspect;
  resize_cuda_image(nspect,1);
  initE(E);
  myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
  int niter = 500;
  double step = 0.6;
  vector<int> sf(ndelay);
  for(int i = 0; i < ndelay; i++) sf[i] = i;
  std::mt19937 mtrnd( std::random_device {} () );
  Real gamma = 1e-3;
  for(int i = 0; i < niter; i++){
    shuffle(sf.begin(),sf.end(), mtrnd);
    double randv = double(rand())/RAND_MAX;
    int k = 0;
    for(int j = 0; j < nfulldelay; j++){
      bool measured = 0;
      int thisdelay = fulldelays[j];
      if(thisdelay == int(delays[k])){
        thisdelay = delays[sf[k]];
        k++;
        measured = 1;
      }
      getMod2((Real*)Eprime, E);
      Real maxv = findMax((Real*)Eprime, nspect);
      dgencTraceSingle(gate, E, trace, thisdelay);
      myFFTM(singleplan, trace, traceprime);
      if(measured) applyModAbs(traceprime, traceIntensity+sf[k-1]*nspect);
      else applySoftThreshold(traceprime, gamma);
      myIFFTM(singleplan, traceprime, traceprime);
      add(traceprime, trace, -1);
      updateGE(E, gate, traceprime, thisdelay, (step)/maxv);
      average(E,gate,0.5);
      if(spectrum){
        myFFTM(singleplan, E, Eprime);
        applyModAbs(Eprime, spectrum);
        applyNorm(Eprime, 1./sqrt(nspect));
        myIFFTM(singleplan, Eprime, E);
        myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
      }
    }
    myFFTM(singleplan, E, Eprime);
    if(spectrum){
      applyModAbs(Eprime, spectrum);
      applyNorm(Eprime, 1./sqrt(nspect));
    }else{
      removeHighFreq(Eprime);
      applyNorm(Eprime, 1./nspect);
    }
    if(i %2 == 0) {
      myIFFTM(singleplan, Eprime, Eprime);
      Real mid = complex<float>(findMiddle(Eprime,nspect)).real();
      shiftmid(Eprime, E, mid*nspect);
    }else{
      myIFFTM(singleplan, Eprime, E);
    }
    myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
  }
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
  int ndelay = 5;
  myDMalloc(Real, delays, ndelay);
  int nspect = 128;
  int nfulldelay = 128;
  myCuDMalloc(Real, d_fulldelays, nfulldelay);
  myDMalloc(Real, fulldelays, nfulldelay);
  myCuDMalloc(Real, d_delays, ndelay);
  myCuDMalloc(complexFormat, d_cE, nspect);
  myCuDMalloc(complexFormat, d_spect, nspect);
  myCuDMalloc(Real, d_spectrum, nspect);
  myCuDMalloc(complexFormat, d_fulltraces, nspect*nfulldelay);
  myCuDMalloc(Real, d_traceIntensity, nspect*ndelay);
  myCuDMalloc(complexFormat, d_traces, nspect*ndelay);
  resize_cuda_image(nspect,1);
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
  myMemcpyD2H(fulldelays, d_fulldelays, nfulldelay*sizeof(Real));
  init_fft(nspect,1,nfulldelay);
  resize_cuda_image(nfulldelay,nspect);
  plt.init(nfulldelay,nspect);
  genTrace(d_cE, d_fulltraces, d_fulldelays);
  convertFOy(d_fulltraces);
  plt.plotComplex(d_fulltraces, MOD2, 0, 1, "trace_truth", 1, 0, 1);
  //select delay, reconstruct E;
  srand(time(NULL));
  for(int i = 0; i < ndelay; i++){
    delays[i] = i*(nfulldelay-1)/(ndelay-1) - nfulldelay/2;// + rand()%10;
  }
  myMemcpyH2D(d_delays, delays, ndelay*sizeof(Real));
  resize_cuda_image(ndelay,nspect);
  plt.init(ndelay,nspect);
  init_fft(nspect,1,ndelay);
  genTrace(d_cE, d_traces, d_delays);
  getMod2(d_traceIntensity, d_traces);
  applyNorm(d_traceIntensity, 1./nspect);
  convertFOy(d_traces);
  plt.plotComplex(d_traces, MOD2, 0, 1, "trace_sampled", 1, 0, 1);
  clearCuMem(d_cE,  nspect*sizeof(complexFormat));
  clearCuMem(d_traces,  nspect*ndelay*sizeof(complexFormat));
  //solveE(d_cE, d_traceIntensity, 0, d_traces, delays, nfulldelay, fulldelays, singleplan);
  solveE(d_cE, d_traceIntensity, d_spectrum, d_traces, delays, nfulldelay, fulldelays, singleplan);
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

