#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <complex.h>
#include "cudaConfig.hpp" //cuda related
#include "cuPlotter.hpp" //plt
#include "cub_wrap.hpp"
#include "fmt/core.h"
#include "fmt/os.h"
#include "misc.hpp"
#include "frog.hpp"
#include "imgio.hpp"
#define c 300
// Units: fs = 1, PHz = 1, nm = 1, c = 300
using namespace std;

void saveWave(const char* fname, Real *x, complexFormat* ccE, int n){
  fmt::ostream file1 = fmt::output_file(fmt::format("{}.txt", fname));
  for(int i = 0; i < n; i++){
    file1.print("{} {} {} {} {}\n", x[i], creal(ccE[i]), cimag(ccE[i]), sq(cabs(ccE[i])), carg(ccE[i]));
  }
  file1.close();
}

void saveWaveSimple(const char* fname, complexFormat* ccE, int n){
  fmt::ostream file1 = fmt::output_file(fmt::format("{}.txt", fname));
  for(int i = 0; i < n; i++){
    file1.print("{} {}\n", creal(ccE[i]), cimag(ccE[i]));
  }
  file1.close();
}

void saveWave(const char* fname, complexFormat* ccE, int n){
  fmt::ostream file1 = fmt::output_file(fmt::format("{}.txt", fname));
  for(int i = 0; i < n; i++){
    file1.print("{} {} {} {} {}\n", i, creal(ccE[i]), cimag(ccE[i]), sq(cabs(ccE[i])), carg(ccE[i]));
  }
  file1.close();
}

void readWave(const char* fname, complexFormat* ccE, int n){
  std::ifstream file1(fname, std::ios::in);
  Real* cE = (Real*) ccE;
  int tmp;
  for(int i = 0; i < n; i++){
    file1>>tmp>>cE[2*i]>>cE[2*i+1];
  }
  file1.close();
}

void genTrace(complexFormat* E, complexFormat* gate, complexFormat* fulltrace, Real* delays, int nspectm = 0){
  dgencTrace(gate, E, fulltrace,delays);
  myFFT(fulltrace, fulltrace);
  applyNorm(fulltrace, 1./sqrt(getCudaCols()));
  convertFOy(fulltrace);
  if(nspectm) zeroEdgey(fulltrace, nspectm);
  convertFOy(fulltrace);
}
void solveE(complexFormat* E, Real* traceIntensity, Real* spectrum, complexFormat* trace, Real* delays, void* singleplan, int nspectm, int nspect, int ndelay, Real* d_delays = 0){
  //myDMalloc(complexFormat, ccE, nspect);
  complexFormat* gate = trace+nspect;
  complexFormat* Eprime = gate+nspect;
  complexFormat* traceprime = Eprime+nspect;
  complexFormat* recTrace;
  Real* recTraceIntensity;
  fmt::ostream *errfile;
  if(d_delays){
    errfile = new fmt::ostream(fmt::output_file("residual.txt"));
    myCuMalloc(complexFormat, recTrace, nspect*ndelay);
    myCuMalloc(Real, recTraceIntensity, nspect*ndelay);
  }
  resize_cuda_image(nspect,1);
  myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
  int niter = 300;
  double step = spectrum?10:2;
  vector<int> sf(ndelay);
  for(int i = 0; i < ndelay; i++) sf[i] = i;
  std::mt19937 mtrnd( std::random_device {} () );
  Real maxv;
  myCuDMalloc(complexFormat, bestE, nspect);
  Real minresidual = 100;
  for(int i = 0; i < niter; i++){
    shuffle(sf.begin(),sf.end(), mtrnd);
    //if(i<30 && i>20) step=0.1;
    if(i==400) {
      step=0.03;
      //myMemcpyD2D(E, bestE, nspect*sizeof(complexFormat));
    }
    myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
    for(int j = 0; j < ndelay; j++){
      double randv = double(rand())/RAND_MAX;
      int thisdelay = delays[sf[j]];
      getMod2((Real*)Eprime, E);
      if(j%10 == 0) maxv = findMax((Real*)Eprime, nspect);
      dgencTraceSingle(gate, E, trace, thisdelay);
      myFFTM(singleplan, trace, traceprime);
      applyModAbsxrange(traceprime, traceIntensity+sf[j]*nspect, 0, nspectm, 1e-2);
      myIFFTM(singleplan, traceprime, traceprime);
      add(traceprime, trace, -1);
      updateGE(E, gate, traceprime, thisdelay, step*(1.*randv/maxv));
      //average(E,gate);
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
    if(i%5==0) {
      myIFFTM(singleplan, Eprime, Eprime);
      initCub();
      Real mid = creal(findMiddle(Eprime,nspect));
      initCub();
      shiftmid(Eprime, E, mid*nspect);
    }else{
      myIFFTM(singleplan, Eprime, E);
    }
    if(d_delays){
      resize_cuda_image(ndelay,nspect);
      genTrace(E,E,recTrace,d_delays);
      getMod2(recTraceIntensity, recTrace);
      int pltn = niter-1;
      if(i == pltn){
        plt.init(ndelay,nspect);
        plt.plotFloat(recTraceIntensity, MOD, 0, 1, "debug", 1, 0, 1);
      }
      add(recTraceIntensity, traceIntensity, -nspect);
      if(i == pltn){
        plt.plotFloat(recTraceIntensity, MOD, 0, 1, "debug1", 1, 0, 1);
        plt.plotFloat(traceIntensity, MOD, 0, nspect, "debug2", 1, 0, 1);
      }
      getMod2(recTraceIntensity, recTraceIntensity);
      Real err = sqrt(findSum(recTraceIntensity)/(nspect*ndelay));
      errfile->print("{} {}\n", i, err);
      if(err < minresidual){
        minresidual = err;
        myMemcpyD2D(bestE, E, nspect*sizeof(complexFormat));
      }
      resize_cuda_image(nspect,1);
    }
  }
  if(d_delays){
    errfile->close();
    delete errfile;
  }
}

void computeTrace(int nfulldelay, Real* d_fulldelays, int nspect, complexFormat* d_cE, complexFormat* d_fulltraces){
  init_fft(nspect,1,nfulldelay);
  resize_cuda_image(nfulldelay,nspect);
  plt.init(nfulldelay,nspect);
  genTrace(d_cE, d_cE, d_fulltraces, d_fulldelays);
  convertFOy(d_fulltraces);
  plt.plotComplex(d_fulltraces, MOD2, 0, 1, "trace_truth", 1, 0, 1);
}

void genE(int nspect, complexFormat* d_cE, complexFormat* ccE){
  resize_cuda_image(nspect,1);
  genEComplex(d_cE);
  myMemcpyD2H(ccE, d_cE, sizeof(complexFormat)*nspect);
  saveWave("input.txt", ccE, nspect);
}
void saveSpect(int nspect, complexFormat* d_cE, Real* d_spectrum, complexFormat* d_spect, complexFormat* ccE, void* singleplan){
  //calculate the spectrum, and write to file
  myFFTM(singleplan, d_cE, d_spect);
  applyNorm(d_spect, 1./sqrt(nspect));
  getMod2(d_spectrum, d_spect);
  cudaConvertFO(d_spect);
  convertFOPhase(d_spect);
  myMemcpyD2H(ccE, d_spect, sizeof(complexFormat)*nspect);
  saveWave("inputSpect.txt", ccE, nspect);
}
void computeDownsampledTrace(int izeroDelay, int ndelay, int nspect, int nspectm, Real* d_delays, Real* d_fulltraceIntensity, Real* d_traceIntensity, int noiseLevel){
  resize_cuda_image(ndelay,nspect);
  downSample(d_traceIntensity, d_fulltraceIntensity, d_delays, nspectm, izeroDelay);
  void* state = newRand(ndelay*nspect);
  initRand(state, time(NULL));
  ccdRecord(d_traceIntensity, d_traceIntensity, noiseLevel, state, 1);
  plt.init(ndelay,nspect);
  plt.plotFloat(d_traceIntensity, MOD, 0, 1, "trace_sampled", 1, 0, 1);
}

//int main(int argc, char** argv )
int main()
{
  init_cuda_image();  //always needed
  bool runSim = 0;
  bool restart = 0;
  Real delay_spacing=0.6;
  Real midlambda =778;
  int noiseLevel = 20;
  Real midfreq = c*2*M_PI/midlambda;
  int freqshift = 0;
  //declare and allocate variables
  int nspect = 1024;
  int nfulldelay = 1024;
  Real* traceExp;
  Real maxfreq,minfreq, freqspacing;
  std::vector<Real> freqs; //freqs stored in PHz, since we have to do FFT, this is used to remap the data to evenly spaced freqs: -512~512
  if(!runSim) {
    //traceExp = readImage("myfrogfreq.bin", nspect, nfulldelay);
    traceExp = readImage("frogdata.bin", nspect, nfulldelay);
    std::ifstream wltable("wavelength.dat");
    Real wavelength;
    while(wltable>>wavelength){
      freqs.push_back(c*2*M_PI/wavelength);
    };
    wltable.close();
    maxfreq = freqs[0], minfreq = freqs.back();
    freqspacing = 2*M_PI/delay_spacing/1024;
    freqshift = (nspect>>1)-(midfreq-minfreq)/freqspacing;
    fmt::println("maxfreq = {:f} PHz, minfreq = {:f} PHz, spacing = {:f}, shift={}", maxfreq, minfreq, freqspacing, freqshift);
  }
  myCuDMalloc(Real, d_fulldelays, nfulldelay);
  myCuDMalloc(complexFormat, d_cE, nspect);
  myCuDMalloc(complexFormat, d_spect, nspect);
  //myCuDMalloc(Real, d_spectrum, nspect);
  myCuDMalloc(complexFormat, d_fulltraces, nspect*nfulldelay);
  myCuDMalloc(Real, d_fulltraceIntensity, nspect*nfulldelay);
  myDMalloc(complexFormat, ccE, nspect);
  resize_cuda_image(nfulldelay,1);
  setDelay(d_fulldelays);

  int ndelay = 400;
  int nspectm= 0;
  Real* delays;
  std::vector<Real> delayv;
  if(runSim) {
    myMalloc(Real, delays, ndelay);
    for(int i = 0; i < ndelay; i++){
      delays[i] = i*Real(nfulldelay-1)/(ndelay-1) - Real(nfulldelay)/2;
    }
  }else{
    std::ifstream delayfile("delays.txt");
    Real tmp;
    int cnt = 0;
    while(delayfile>>tmp){
      if(cnt++%20 !=0) continue;
      if(tmp > (nfulldelay>>1)) break;
      delayv.push_back(tmp);  // fs to # of pix, pix size is the temporal resolution of electric field, determined by the frequency range.
    }
    delayfile.close();
    ndelay = delayv.size();
    delays = delayv.data();
  }
  myCuDMalloc(Real, d_delays, ndelay);
  myMemcpyH2D(d_delays, delays, ndelay*sizeof(Real));

  myCuDMalloc(Real, d_traceIntensity, nspect*ndelay);
  myCuDMalloc(complexFormat, d_traces, nspect*ndelay);
  myCuDMalloc(Real, d_traceLambda, nspect*nfulldelay);
  myCuDMalloc(Real, d_freqs, nspect);
  //Generate electric field, and write to file
  void* singleplan;
  createPlan1d(&singleplan, nspect);
  if(runSim){
    genE(nspect, d_cE, ccE);
    //saveSpect(nspect, d_cE, d_spectrum, d_spect, ccE, singleplan);
    //calculate the complete FROG trace and plot
    computeTrace(nfulldelay, d_fulldelays, nspect, d_cE, d_fulltraces);
    clearCuMem(d_cE,  nspect*sizeof(complexFormat));
    getMod2(d_fulltraceIntensity, d_fulltraces);
    plt.saveFloat(d_fulltraceIntensity, "simfrogdata");
    //downsampling, calculate downsampled trace and plot.
    computeDownsampledTrace(nfulldelay, ndelay, nspect, nspectm, d_delays, d_fulltraceIntensity, d_traceIntensity, noiseLevel);
  }else{
    myMemcpyH2D(d_freqs, freqs.data(), sizeof(Real)*nspect);
    if(restart){
      readWave("output_short.txt", ccE, nspect);
      myMemcpyH2D(d_cE, ccE, sizeof(complexFormat)*nspect);
    }
    if(1){
      myMemcpyH2D(d_traceLambda, traceExp, nspect*nfulldelay*sizeof(Real));
      resize_cuda_image(nfulldelay,nspect);
      plt.init(nfulldelay, nspect);
      plt.plotFloat(d_traceLambda, MOD, 0, 1, "inputimg.png", 1, 0, 1);
      resize_cuda_image(nfulldelay, 1);
      traceLambdaToFreq(d_fulltraceIntensity, d_traceLambda, d_freqs, nspect, minfreq, freqspacing, freqshift);
      resize_cuda_image(nfulldelay,nspect);
      applyNorm(d_fulltraceIntensity, 1./findMax(d_fulltraceIntensity));
      plt.plotFloat(d_fulltraceIntensity, MOD, 0, 1, "inputtrace", 1, 0, 1);
      computeDownsampledTrace(508, ndelay, nspect, nspectm, d_delays, d_fulltraceIntensity, d_traceIntensity, noiseLevel);
      resize_cuda_image(ndelay,nspect);
      plt.init(ndelay, nspect);
      plt.saveFloat(d_traceIntensity,"myfrogfreq");
    }else{
      myMemcpyH2D(d_traceIntensity, traceExp, nspect*nfulldelay*sizeof(Real));
      //myMemcpyH2D(d_fulltraceIntensity, traceExp, nspect*nfulldelay*sizeof(Real));
      resize_cuda_image(ndelay,nspect);
      plt.init(ndelay,nspect);
      //computeDownsampledTrace(nfulldelay, ndelay, nspect, nspectm, d_delays, d_fulltraceIntensity, d_traceIntensity, noiseLevel);
    }
  }
  //Reconstruct electric field
  clearCuMem(d_traces,  nspect*ndelay*sizeof(complexFormat));
  applyNorm(d_traceIntensity, 1./nspect);
  convertFOy(d_traceIntensity);
  if(!restart){
    resize_cuda_image(nspect,1);
    initE(d_cE);
  }
  init_fft(nspect,1,ndelay);
  solveE(d_cE, d_traceIntensity, 0, d_traces, delays, singleplan, nspectm, nspect, ndelay, d_delays); // spectrum is unknown
  //solveE(d_cE, d_traceIntensity, d_spectrum, d_traces, delays, singleplan, nspectm); //spectrum is known

  //save electric field to file
  myMemcpyD2H(ccE, d_cE, sizeof(complexFormat)*nspect);
  myDMalloc(Real, xaxis, nspect);
  for(int i = 0; i < nspect; i++){
    xaxis[i] = i*delay_spacing;
  }
  saveWave("output", xaxis, ccE, nspect);
  saveWaveSimple("output_short", ccE, nspect);
  //save spectrum to file
  myFFTM(singleplan, d_cE, d_spect);
  resize_cuda_image(nspect,1);
  applyNorm(d_spect, 1./sqrt(nspect));
  cudaConvertFO(d_spect);
  convertFOPhase(d_spect);
  myMemcpyD2H(ccE, d_spect, sizeof(complexFormat)*nspect);
  int nsavespect = 0;
  for(int i = 0; i < nspect; i++){
    Real freq = (i - Real(nspect)/4-Real(freqshift)/2)*freqspacing + minfreq/2;
    if(freq < minfreq/2) continue;
    xaxis[nsavespect] = 2*M_PI*c / freq;
    ccE[nsavespect] = ccE[i];
    nsavespect++;
  }
  saveWave("outputSpect", xaxis, ccE, nsavespect);
  //calculate reconstructed complete trace
  init_fft(nspect,1,nfulldelay);
  resize_cuda_image(nfulldelay,nspect);
  plt.init(nfulldelay,nspect);
  genTrace(d_cE, d_cE, d_fulltraces, d_fulldelays);
  convertFOy(d_fulltraces);
  plt.plotComplex(d_fulltraces, MOD2, 0, 1, "trace_recon_full", 1, 0, 1);
  return 0;
}
