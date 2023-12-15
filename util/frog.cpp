#include "cudaConfig.h" //cuda related
#include "cuPlotter.h" //plt
#include <complex.h>
#include "cub_wrap.h"
#include "frog.h"
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

void saveWave(const char* fname, complexFormat* ccE, int n){
  std::ofstream file1(fname, std::ios::out);
  for(int i = 0; i < n; i++){
    file1<<i << " " << creal(ccE[i]) << " " << cimag(ccE[i]) << " " << cabs(ccE[i]) << " " << carg(ccE[i])<<std::endl;
  }
  file1.close();
}
void solveE(complexFormat* E, Real* traceIntensity, Real* spectrum, complexFormat* trace, Real* delays, int nfulldelay, int singleplan, int nspectm){
  int ndelay = getCudaRows();
  int nspect = getCudaCols();
  //myDMalloc(complexFormat, ccE, nspect);
  complexFormat* gate = trace+nspect;
  complexFormat* Eprime = gate+nspect;
  complexFormat* traceprime = Eprime+nspect;
  resize_cuda_image(nspect,1);
  initE(E);
  myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
  int niter = 3000;
  double step = spectrum?10:3.;
  vector<int> sf(ndelay);
  for(int i = 0; i < ndelay; i++) sf[i] = i;
  std::mt19937 mtrnd( std::random_device {} () );
  for(int i = 0; i < niter; i++){
    shuffle(sf.begin(),sf.end(), mtrnd);
    for(int j = 0; j < ndelay; j++){
      double randv = double(rand())/RAND_MAX;
      int thisdelay = delays[sf[j]];
      getMod2((Real*)Eprime, E);
      Real maxv = findMax((Real*)Eprime, nspect);
      dgencTraceSingle(gate, E, trace, thisdelay);
      myFFTM(singleplan, trace, traceprime);
      applyModAbsxrange(traceprime, traceIntensity+sf[j]*nspect, 0, nspectm, 1e-3);
      myIFFTM(singleplan, traceprime, traceprime);
      add(traceprime, trace, -1);
      updateGE(E, gate, traceprime, thisdelay, (step*randv)/maxv);
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
    if(i %10 == 0) {
      myIFFTM(singleplan, Eprime, Eprime);
      Real mid = creal(findMiddle(Eprime,nspect));
      shiftmid(Eprime, E, mid*nspect);
    }else{
      myIFFTM(singleplan, Eprime, E);
    }
    myMemcpyD2D(gate, E, nspect*sizeof(complexFormat));
  }
}

void genTrace(complexFormat* E, complexFormat* gate, complexFormat* fulltrace, Real* delays, int nspectm = 0){
  dgencTrace(gate, E, fulltrace,delays);
  myFFT(fulltrace, fulltrace);
  applyNorm(fulltrace, 1./sqrt(getCudaCols()));
  convertFOy(fulltrace);
  if(nspectm) zeroEdgey(fulltrace, nspectm);
  convertFOy(fulltrace);
}

void genTrace(complexFormat* E, complexFormat* fulltrace, Real* delays, int nspectm = 0){
  dgencTrace(E, E, fulltrace,delays);
  myFFT(fulltrace, fulltrace);
  applyNorm(fulltrace, 1./sqrt(getCudaCols()));
  convertFOy(fulltrace);
  if(nspectm) zeroEdgey(fulltrace, nspectm);
  convertFOy(fulltrace);
}

int main(int argc, char** argv )
{
  init_cuda_image();  //always needed
  int ndelay = 5;
  int nspect = 128;
  int nspectm=58;
  int nfulldelay = 128;
  int noiseLevel = 20;
  //declare and allocate variables
  myDMalloc(Real, delays, ndelay);
  myCuDMalloc(Real, d_fulldelays, nfulldelay);
  myCuDMalloc(Real, d_delays, ndelay);
  myCuDMalloc(complexFormat, d_cE, nspect);
  myCuDMalloc(complexFormat, d_spect, nspect);
  myCuDMalloc(Real, d_spectrum, nspect);
  myCuDMalloc(complexFormat, d_fulltraces, nspect*nfulldelay);
  myCuDMalloc(Real, d_traceIntensity, nspect*ndelay);
  myCuDMalloc(complexFormat, d_traces, nspect*ndelay);
  //Generate electric field, and write to file
  resize_cuda_image(nspect,1);
  genEComplex(d_cE);
  myDMalloc(complexFormat, ccE, nspect);
  myMemcpyD2H(ccE, d_cE, sizeof(complexFormat)*nspect);
  saveWave("input.txt", ccE, nspect);

  //calculate the spectrum, and write to file
  int singleplan;
  createPlan1d(&singleplan, nspect);
  myFFTM(singleplan, d_cE, d_spect);
  applyNorm(d_spect, 1./sqrt(nspect));
  getMod2(d_spectrum, d_spect);
  cudaConvertFO(d_spect);
  convertFOPhase(d_spect);
  myMemcpyD2H(ccE, d_spect, sizeof(complexFormat)*nspect);
  saveWave("inputSpect.txt", ccE, nspect);
  //calculate the complete FROG trace and plot
  resize_cuda_image(nfulldelay,1);
  setDelay(d_fulldelays);
  init_fft(nspect,1,nfulldelay);
  resize_cuda_image(nfulldelay,nspect);
  plt.init(nfulldelay,nspect);
  genTrace(d_cE, d_fulltraces, d_fulldelays);
  convertFOy(d_fulltraces);
  plt.plotComplex(d_fulltraces, MOD2, 0, 1, "trace_truth", 1, 0, 1);
  //downsampling, calculate downsampled trace and plot.
  for(int i = 0; i < ndelay; i++){
    delays[i] = i*Real(nfulldelay-1)/(ndelay-1) - Real(nfulldelay)/2;
  }
  myMemcpyH2D(d_delays, delays, ndelay*sizeof(Real));
  resize_cuda_image(ndelay,nspect);
  void* state = newRand(ndelay*nspect);
  initRand(state, time(NULL));
  plt.init(ndelay,nspect);
  init_fft(nspect,1,ndelay);
  genTrace(d_cE, d_traces, d_delays, nspectm);
  getMod2(d_traceIntensity, d_traces);
  applyNorm(d_traceIntensity, 1./nspect);
  ccdRecord(d_traceIntensity, d_traceIntensity, noiseLevel, state, 1);
  convertFOy(d_traces);
  plt.plotComplex(d_traces, MOD2, 0, 1, "trace_sampled", 1, 0, 1);
  //Reconstruct electric field
  clearCuMem(d_cE,  nspect*sizeof(complexFormat));
  clearCuMem(d_traces,  nspect*ndelay*sizeof(complexFormat));
  //solveE(d_cE, d_traceIntensity, 0, d_traces, delays, nfulldelay, singleplan, nspectm); // spectrum is unknown
  solveE(d_cE, d_traceIntensity, d_spectrum, d_traces, delays, nfulldelay, singleplan, nspectm); //spectrum is known

  //save electric field to file
  myMemcpyD2H(ccE, d_cE, sizeof(complexFormat)*nspect);
  saveWave("output.txt", ccE, nspect);
  //save spectrum to file
  myFFTM(singleplan, d_cE, d_spect);
  resize_cuda_image(nspect,1);
  applyNorm(d_spect, 1./sqrt(nspect));
  cudaConvertFO(d_spect);
  convertFOPhase(d_spect);
  myMemcpyD2H(ccE, d_spect, sizeof(complexFormat)*nspect);
  saveWave("outputSpect.txt", ccE, nspect);
  //calculate reconstructed complete trace
  init_fft(nspect,1,nfulldelay);
  resize_cuda_image(nfulldelay,nspect);
  plt.init(nfulldelay,nspect);
  genTrace(d_cE, d_fulltraces, d_fulldelays);
  convertFOy(d_fulltraces);
  plt.plotComplex(d_fulltraces, MOD2, 0, 1, "trace_recon_full", 1, 0, 1);
}
