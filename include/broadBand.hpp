#include "format.hpp"
class broadBand{
  public:
    int *locplan;
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
    broadBand(){};
    void calcPixelWeights();
    void writeSpectra(const char* filename);
    void resetSpectra();
    void init(int nrow, int ncol, double minlambda, double maxlambda);
    void init(int nrow, int ncol, int nlambda_, double* lambdas_, double* spectra_);
    Real init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray);
    void generateMWL(void* d_input, void* d_patternSum, void* single = 0);
};
