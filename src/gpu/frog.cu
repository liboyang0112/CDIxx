#include "cudaDefs_h.cu"
#include "cuComplex.h"
#include <curand_kernel.h>

cuFunc(dgencTraceSingle,(complexFormat* gate, complexFormat* E, complexFormat* trace, Real delay),(gate, E, trace,delay), {
  cuda1Idx();
  int tidx = index-delay;
  if(tidx >= cuda_row || tidx < 0) {
    trace[index].x = 0;
    trace[index].y = 0;
  }else
    trace[index] = cuCmulf(gate[tidx],E[index]);
})

cuFunc(dgencTrace,(complexFormat* gate, complexFormat* E, complexFormat* fulltrace, Real* delay),(gate, E, fulltrace,delay), {
  cudaIdx();
  int tidx = y-delay[x];
  if(tidx >= cuda_column || tidx < 0) {
    fulltrace[index].x = 0;
    fulltrace[index].y = 0;
  }else
    fulltrace[index] = cuCmulf(gate[tidx],E[y]);
})


cuFunc(genEComplex,(complexFormat* E),(E),{
  cuda1Idx();
  int bias = index-cuda_row/2;
  Real sigma = 50;
  Real midf = 0;
  Real chirp = 4e-4;
  Real CEP = M_PI;
  Real phase = 2*M_PI*midf*index + CEP + 2*M_PI*chirp*bias*bias;
  Real envolope = (exp(-sq(bias-100)/(2*sq(sigma)))+0.5*exp(-sq(bias+100)/(2*sq(sigma))));
  Real c, s;
  sincosf(phase, &s, &c);
  E[index].x = envolope*c;
  E[index].y = envolope*s;
})

cuFunc(setDelay,(Real* delay),(delay),{
  cuda1Idx();
  delay[index] = index-cuda_row/2;
})

cuFunc(convertFOy,(complexFormat* data),(data), {
  cuda1Idx();
  int y = index%cuda_column;
  if(y >= cuda_column/2) return;
  cuComplex tmp = data[index];
  data[index] = data[index+cuda_column/2];
  data[index+cuda_column/2] = tmp;
})
cuFunc(convertFOy, (Real* data),(data), {
  cuda1Idx();
  int y = index%cuda_column;
  if(y >= cuda_column/2) return;
  Real tmp = data[index];
  data[index] = data[index+cuda_column/2];
  data[index+cuda_column/2] = tmp;
})
cuFunc(updateGE,(complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha),(E,gate,trace,delay, alpha), {
  cuda1Idx();
  int tidx = index - delay;
  if(tidx >= cuda_row || tidx < 0) return;
  //if(index < cuda_row/4 || index > 3*cuda_row/4) {
  //  E[index].x = E[index].y = 0;
  //  return;
  //}
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
cuFunc(updateE,(complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha),(E,gate,trace,delay, alpha), {
  cuda1Idx();
  int tidx = index - delay;
  if(tidx >= cuda_row || tidx < 0) return;
  cuComplex tmp = gate[tidx];
  tmp.y = -tmp.y;
  tmp = cuCmulf(tmp,trace[index]);
  E[index].x += alpha*tmp.x;
  E[index].y += alpha*tmp.y;
})

cuFunc(initE,(complexFormat* gate),(gate), {
  cuda1Idx();
  int window = 32;
  if(index > cuda_row/2-window && index < cuda_row/2+window){
    gate[index].x = 1;
  }else gate[index].x = 0;
  gate[index].y = 0;
})

cuFunc(removeHighFreq,(complexFormat* data),(data), {
  cuda1Idx();
  if(index >= cuda_row/5*2 && index < cuda_row/5*3){
    data[index].x = 0;
    data[index].y = 0;
  }
})

cuFunc(shiftmid,(complexFormat* Eprime, complexFormat* E, int tx),(Eprime, E, tx),{
  cuda1Idx();
  int tidx = index+tx;
  if(tidx < 0 || tidx >= cuda_row) {
    E[index].x = 0;
    E[index].y = 0;
  }
  else E[index] = Eprime[tidx];
})

cuFunc(average,(complexFormat* Eprime, complexFormat* E),(Eprime, E),{
  cuda1Idx();
  Eprime[index].x = E[index].x = (Eprime[index].x+E[index].x)/2;
  Eprime[index].y = E[index].y = (Eprime[index].y+E[index].y)/2;
})
cuFunc(applyModAbsxrange,(complexFormat* source, Real* target, void* state, int xrange, Real gamma),(source, target, state, xrange, gamma),{
    cuda1Idx();
    Real mod = hypot(source[index].x, source[index].y);
    if(index < xrange+cuda_row/2 && index >= cuda_row/2 - xrange){
      Real thres = gamma*sqrtf(cuda_row);
      if(mod > thres){
        mod = (1-thres/mod)/cuda_row;
        source[index].x *= mod;
        source[index].y *= mod;
      }else source[index].x = source[index].y = 0;
      return;
    }
    Real rat = target[index];
    if(rat > 0) rat = sqrt(rat);
    else {
      source[index].x = source[index].y = 0;
      return;
    }
    if(abs(mod)<1e-5) {
      if(rat > 1e-3) {
        Real randphase = state?curand_uniform((curandStateMRG32k3a*)state + index)*2*M_PI:0;
        Real c, s;
        sincosf(randphase, &s, &c);
        source[index].x = rat*c;
        source[index].y = rat*s;
      }
      return;
    }
    else rat /= mod;
    source[index].x *= rat;
    source[index].y *= rat;
    })
cuFunc(downSample, (Real* out, Real* input, Real* colsel, int rowcut, int midplace), (out, input, colsel, rowcut, midplace), {
    cudaIdx();
    if(y < rowcut || y >= cuda_column - rowcut){
      out[index] = 0;
    }else{
      out[index] = input[int(colsel[x]+midplace)*cuda_column+y];
    }
    })
cuFunc(traceLambdaToFreq, (Real* d_traceIntensity, Real* d_traceLambda, Real* d_freqs, int nspect, Real minfreq, Real freqspacing, int freqshift), (d_traceIntensity, d_traceLambda, d_freqs, nspect, minfreq, freqspacing, freqshift), {
    cuda1Idx();
    int idx = index*nspect;
    Real c_freq = minfreq, freq1, freq2, a;
    int cnt = 1;
    while(freqshift < nspect){
        while(d_freqs[nspect-cnt-1] < c_freq) cnt++;
        freq1 = d_freqs[nspect-cnt];
        freq2 = d_freqs[nspect-cnt-1];
        a = (c_freq-freq1)/(freq2-freq1);
        d_traceIntensity[idx+freqshift] = ((1-a)*d_traceLambda[idx+nspect-cnt-1] + a*d_traceLambda[idx+nspect-cnt])/sq(c_freq);
        c_freq += freqspacing;
        freqshift++;
    }
    })
