#include "format.h"
class monoChromo{
  public:
    void *locplan;
    void **gs=0;
    void *gr=0;
    void *crlt_gs_gr=0;
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
    void calcPixelWeights();
    void writeSpectra(const char* filename);
    void resetSpectra();
    void init(int nrow, int ncol, int nlambda_, double* lambdas_, double* spectra_);
    Real init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray);
    void generateMWL(void* d_input, void* d_patternSum, void* single = 0, Real oversampling = 2);
    void solveMWL(void* d_input, void* d_patternSum, bool restart = 0, int nIter = 200, bool updateX = 1, bool updateY = 0);
};
