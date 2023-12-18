#include "broadBand.hpp"
class spectPhase: public broadBand{
  int pixCount;
  complexFormat* d_ref;  //full image
  uint32_t* d_supportMap; //mapping the support
  complexFormat* d_support; //compressed support
  complexFormat* cspectrum;
  public:
    spectPhase():broadBand(){};
    void initRefSupport(complexFormat* refer, complexFormat* support);
    void solvecSpectrum(Real* pattern, int niter);
    void generateMWL(void* pattern, void* mat, Real thickness);
};
