#include "cudaConfig.hpp" //cuda related
#include "cudaDefs.hpp"
#include <curand_kernel.h>
#include "frog.hpp"

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
cuFuncc(updateGE, (complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha),(cuComplex* E, cuComplex* gate, cuComplex* trace, Real delay, Real alpha), ((cuComplex*)E,(cuComplex*)gate,(cuComplex*)trace,delay, alpha), {
  cuda1Idx();
  int tidx = index - delay;
  if(tidx >= cuda_row || tidx < 0) return;
  if(index < cuda_row/4 || index > 3*cuda_row/4) {
    E[index].x = E[index].y = 0;
    return;
  }
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
cuFuncc(applyModAbsxrange,(complexFormat* source, Real* target, void* state, int xrange, Real gamma),(cuComplex* source, Real* target, void* state, int xrange, Real gamma),((cuComplex*)source, target, state, xrange, gamma),{
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
    else rat = 0;
    if(mod==0) {
    Real randphase = state?curand_uniform((curandStateMRG32k3a*)state + index)*2*M_PI:0;
    source[index].x = rat*cos(randphase);
    source[index].y = rat*sin(randphase);
    return;
    }
    rat /= mod;
    source[index].x *= rat;
    source[index].y *= rat;
    })
