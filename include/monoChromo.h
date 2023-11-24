#include "format.h"
class monoChromo{
  public:
    int *locplan;
    void **gs=0;  //sample, for each wavelength
    void **refs=0;  //reference data, for each wavelength
    void *d_maskMap = 0; //reference data map. mapping data into masks. on device
    void *d_refMask=0;  //reference mask. on device
    int pixCount = 0;
    bool useOrth = 0;
    double *spectra;  //normalized spectra
    double *lambdas;  //normalized lambdas, 1 is the shortest
    Real *pixel_weight;
    int jump = 20;
    int skip = 10;
    int nlambda;
    int *rows;
    int *cols;
    int row;
    int column;
    void *devstates = 0;
    monoChromo(){};
    void initRefs(const char* maskFile);
    void clearRefs();
    void assignRef(void* wavefront, int i);
    void assignRef(void* wavefront);
    void generateMWLRefPattern(void* d_patternSum, bool debug = 0);
    void reconRefs(void* d_patternSum);
    void calcPixelWeights();
    void writeSpectra(const char* filename);
    void resetSpectra();
    void init(int nrow, int ncol, double minlambda, double maxlambda);
    void init(int nrow, int ncol, int nlambda_, double* lambdas_, double* spectra_);
    Real init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray);
    void generateMWL(void* d_input, void* d_patternSum, void* single = 0);
    void solveMWL(void* d_input, void* d_patternSum, int noiseLevel = 0, bool restart = 0, int nIter = 200, bool updateX = 1, bool updateY = 0);
};
