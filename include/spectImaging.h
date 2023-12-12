#include "broadBand.h"
class spectImaging : public broadBand{
  public:
    void **refs=0;  //reference data, for each wavelength
    void *d_maskMap = 0; //reference data map. mapping data into masks. on device
    void *d_refMask=0;  //reference mask. on device
    int pixCount = 0;
    spectImaging():broadBand(){};
    void initRefs(const char* maskFile);
    void clearRefs();
    void assignRef(void* wavefront, int i);
    void assignRef(void* wavefront);
    void generateMWLRefPattern(void* d_patternSum, bool debug = 0);
    void reconRefs(void* d_patternSum);
};
