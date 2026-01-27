#include "format.hpp"
void applySupport(Real* image, Real* support);
void multiplyProbe(complexFormat* object, complexFormat* probe, complexFormat* U, int shiftx, int shifty, int objrow, int objcol, complexFormat *window = 0);
void getWindow(complexFormat* object, int shiftx, int shifty, int objrow, int objcol, complexFormat *window);
void updateWindow(complexFormat* object, int shiftx, int shifty, int objrow, int objcol, complexFormat *window);
void addWindow(complexFormat* object, int shiftx, int shifty, int objrow, int objcol, complexFormat *window, Real norm = 1);
void updateObject(complexFormat* object, complexFormat* probe, complexFormat* U, Real mod2maxProbe);
void updateObjectAndProbe(complexFormat* object, complexFormat* probe, complexFormat* U, Real mod2maxProbe, Real mod2maxObj);
void updateObjectAndProbeStep(complexFormat* object, complexFormat* probe, complexFormat* probeStep, complexFormat* U, Real mod2maxProbe, Real mod2maxObj, Real stepsize);
void updateObjectStepAndProbeStep(complexFormat* object, complexFormat* probe, complexFormat* probeStep, complexFormat* U, Real mod2maxProbe, Real mod2maxObj, Real norm = 1);
void random(complexFormat* object, void *state);
void pupilFunc(complexFormat* object);
void multiplyx(complexFormat* object, complexFormat* out);
void multiplyy(complexFormat* object, complexFormat* out);
void calcPartial(Real* out, complexFormat* object, complexFormat* Fn, Real* pattern, Real* beamstop);
void applySupport(Real* image, Real* support);
