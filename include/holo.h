#include "cdi.h"
class holo : public CDI{
  public:
    holo(const char* configfile);
    Real *patternData_holo = 0;
    Real *patternData_obj = 0;
    Real *xcorrelation = 0;
    complexFormat *patternWave_holo = 0;
    complexFormat *patternWave_obj = 0;
    complexFormat *objectWave_holo = 0;
    Real *support_holo = 0;
    Real *xcorrelation_support = 0; //two times the size of original support
    int objrow, objcol;
    void calcXCorrelation();
    void simulate();
    void allocateMem_holo();
    void iterate();
};
