#include "format.h"
class monoChromo{
  public:
    void *locplan;
    Real *spectra;  //normalized spectra
    Real *lambdas;  //normalized lambdas, 1 is the shortest
    int nlambda;
    int *rows;
    int *cols;
    int row;
    int column;
    monoChromo(){};
    void writeSpectra(const char* filename);
    void resetSpectra();
    void init(int nrow, int ncol, int nlambda_, Real* lambdas_, Real* spectra_);
    void init(int nrow, int ncol, Real* lambdasi, Real* spectrumi, Real endlambda);
    void generateMWL(void* d_input, void* d_patternSum, void* single = 0, Real oversampling = 2);
    void solveMWL(void* d_input, void* d_patternSum, bool restart = 0, int nIter = 200, bool updateX = 1, bool updateY = 0);
};
