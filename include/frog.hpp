#include "format.hpp"
void dgencTraceSingle (complexFormat* gate, complexFormat* E, complexFormat* trace, Real delay);
void dgencTrace (complexFormat* gate, complexFormat* E, complexFormat* fulltrace, Real* delay);
void genEComplex(complexFormat* E);
void convertFOy (complexFormat* data);
void setDelay(Real* delay);
void updateGE (complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha);
void updateE (complexFormat* E, complexFormat* gate, complexFormat* trace, Real delay, Real alpha);
void initE (complexFormat* gate);
void removeHighFreq (complexFormat* data);
void shiftmid (complexFormat* Eprime, complexFormat* E, int tx);
void average (complexFormat* Eprime, complexFormat* E, Real gamma);
void applyModAbsxrange(complexFormat* source, Real* target, void* state, int xrange, Real gamma);
