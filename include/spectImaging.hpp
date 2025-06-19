#include "broadBand.hpp"
class spectImaging : public broadBand{
  public:
    void **refs=0;  //reference data, for each wavelength
    void **spectImages=0;  //spectral image, same size for each wavelength, small images, use paste function to combine with refs
    uint32_t *d_maskMap = 0; //reference data map. mapping data into masks. on device
    void *d_refMask=0;  //reference mask. on device
    int pixCount = 0;
    int imrow, imcol;
    spectImaging():broadBand(){};
    void initRefs(Real* mask, int row, int col, int shiftx = 0, int shifty = 0);
    void pointRefs(int npoints, int *xs, int *ys);
    void initHSI(int row, int col);
    void saveHSI(const char* name, Real* support);
    void clearRefs();
    void assignRef(void* wavefront, int i);
    void assignRef(void* wavefront);
    void generateMWLRefPattern(void* d_patternSum, bool debug = 0);
    void generateMWLPattern(void* d_patternSum, bool debug = 0, Real* mask = 0);
    void reconRefs(void* d_patternSum);
    void clearHSI();
    void reconstructHSI(void* d_patternSum, Real* mask);
};
