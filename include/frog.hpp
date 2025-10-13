#include "format.hpp"
void dgencTraceSingle (complexFormat* gate, complexFormat* E, complexFormat* trace, Real delay);
void dgencTrace (complexFormat* gate, complexFormat* E, complexFormat* fulltrace, Real* delay);
void genEComplex(complexFormat* E);
void convertFOy (complexFormat* data);
void convertFOy (Real* data);
void setDelay(Real* delay);
void updateGE (complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha);
void updateE (complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha);
void initE (complexFormat* gate);
void removeHighFreq (complexFormat* data);
void shiftmid (complexFormat* Eprime, complexFormat* E, int tx);
void average (complexFormat* Eprime, complexFormat* E);
void applyModAbsxrange(complexFormat* source, Real* target, void* state, int xrange, Real gamma);
void downSample (Real* out, Real* input, Real* colsel, int rowcut, int nfulldelay);
void traceLambdaToFreq (Real* d_traceIntensity, Real* d_traceLambda, Real* d_freqs, int nspect, Real minfreq, Real freqspacing, int freqshift);
