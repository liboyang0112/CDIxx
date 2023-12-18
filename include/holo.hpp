#include "cdi.hpp"
class holo : public CDI{
  public:
    holo(const char* configfile);
    Real *patternData_holo = 0;
    Real *patternData_obj = 0;
    Real *xcorrelation = 0;
    void *patternWave_holo = 0;
    void *patternWave_obj = 0;
    void *objectWave_holo = 0;
    Real *support_holo = 0;
    Real *xcorrelation_support = 0; //two times the size of original support
    int objrow, objcol;
    void initXCorrelation();
    void calcXCorrelation(bool doplot);
    void calcXCorrelationHalf(bool doplot);
    void simulate();
    void allocateMem_holo();
    void iterate();
    void noniterative();
};
