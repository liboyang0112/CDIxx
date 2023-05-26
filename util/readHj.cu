#include"cnpy.h"
#include"cudaConfig.h"
#include"format.h"
#include"cuPlotter.h"
#include"common.h"
#include<complex>
#include<cstdlib>
#include<iostream>
#include<random>
#include<chrono>
#include<map>
#include<string>
#include<fstream>
#include"monoChromo.h"
#include"cub_wrap.h"

int main(int argc, const char* argv[])
{
    cnpy::npz_t my_npz = cnpy::npz_load("Example_whitelight_experiment.npz"); //arrays are saved in double
    cnpy::NpyArray b = my_npz["B"];
    cnpy::NpyArray spectra = my_npz["spect_I"];
    cnpy::NpyArray lambdas = my_npz["spect_lambda"];
    printf("numpy word size= %ld, double=%ld\n", b.word_size, sizeof(double));
    int row = b.shape[0];
    int col = b.shape[1];
    int nlambda = lambdas.shape[0];
    double* spectrad = spectra.data<double>();
    double* lambdasd = lambdas.data<double>();
    for(int i = 1; i < nlambda; i++)
      lambdasd[i] /= lambdasd[0];
    lambdasd[0] = 1;
    printf("image size = (%d, %d), spectra size = %d\n", row, col, nlambda);

    Real* realb = (Real*)memMngr.borrowCache(sizeof(Real)*row*col);
    double* doubleb = (double*)memMngr.useOnsite(sizeof(double)*row*col);
    cudaMemcpy(doubleb, b.data<double>(), sizeof(double)*row*col, cudaMemcpyHostToDevice);
    init_cuda_image(row, col);
    cudaF(assignVal,realb, doubleb);
    cudaF(applyNorm,realb, 1./findMax(realb));
    monoChromo mwl;
    mwl.jump = 25;
    mwl.skip = 20;
    mwl.init(row, col, lambdasd, spectrad, nlambda);
    plt.init(row, col);
    complexFormat* complexpattern = (complexFormat*)memMngr.borrowCache(sizeof(double)*row*col);
    complexFormat* solved = (complexFormat*)memMngr.borrowCache(sizeof(double)*row*col);
    cudaF(extendToComplex,realb, complexpattern);
    plt.saveFloat(realb, "broadpattern");
    plt.plotComplex(complexpattern,REAL,0,1,"logbroadpattern",1);
    plt.plotComplex(complexpattern,REAL,0,1,"broadpattern",0);
    printf("solving matrix\n");
    mwl.solveMWL(complexpattern, solved, 1, 40, 1, 0);
    plt.plotComplex(solved,REAL,0,1,"logmonopattern",1);
    plt.plotComplex(solved,REAL,0,1,"monopattern",0);
    cudaF(getMod,realb, solved);
    plt.saveFloat(realb, "pattern");
    myCufftExec( *plan, complexpattern, complexpattern, CUFFT_INVERSE);
    cudaF(applyNorm,complexpattern,1./col);
    plt.plotComplex(complexpattern, MOD, 1, 1, "autocbroad", 1);
    myCufftExec( *plan, solved, complexpattern, CUFFT_INVERSE);
    cudaF(applyNorm,complexpattern,1./col);
    plt.plotComplex(complexpattern, MOD, 1, 1, "autocsolved", 1);
    mwl.writeSpectra("spectra_new.txt");

    return 0;
}
