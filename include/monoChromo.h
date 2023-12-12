#include "format.h"
#include "broadBand.h"
class monoChromo : public broadBand{
  public:
    void **gs=0;  //sample, for each wavelength
    bool useOrth = 0;
    monoChromo():broadBand(){};
    void solveMWL(void* d_input, void* d_patternSum, int noiseLevel = 0, bool restart = 0, int nIter = 200, bool updateX = 1, bool updateY = 0);
};
