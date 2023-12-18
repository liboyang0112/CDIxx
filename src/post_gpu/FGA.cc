#include"cudaConfig.hpp"
#include"FGA.hpp"
#include"cuPlotter.hpp"
#include<fstream>
#include"monoChromo.hpp"
#include"cub_wrap.hpp"

int FGA(int row, int col, int nlambda, double* lambdas, double* spectra, Real* data)
{
    printf("image size = (%d, %d), spectra size = %d\n", row, col, nlambda);
    Real* realb = (Real*)memMngr.borrowCache(sizeof(Real)*row*col);
    myMemcpyH2D(realb, data, sizeof(Real)*row*col);
    init_cuda_image();
    resize_cuda_image(row, col);
    applyNorm(realb, 1./findMax(realb));
    monoChromo mwl;
    mwl.jump = 10;
    mwl.skip = 5;
    mwl.init(row, col, lambdas, spectra, nlambda);
    plt.init(row, col);
    complexFormat* complexpattern = (complexFormat*)memMngr.borrowCache(sizeof(double)*row*col);
    complexFormat* solved = (complexFormat*)memMngr.borrowCache(sizeof(double)*row*col);
    extendToComplex(realb, complexpattern);
    plt.saveFloat(realb, "broadpattern");
    plt.plotComplex(complexpattern,MOD,0,1,"logbroadpattern",1,0,1);
    plt.plotComplex(complexpattern,MOD,0,1,"broadpattern",0);
    printf("solving matrix\n");
    mwl.solveMWL(complexpattern, solved, 0, 1, 80, 1, 0);
    plt.plotComplex(solved,MOD,0,1,"logmonopattern",1, 0, 1);
    plt.plotComplex(solved,MOD,0,1,"monopattern",0);
    getMod(realb, solved);
    plt.saveFloat(realb, "pattern");
    myIFFT(complexpattern, complexpattern);
    applyNorm(complexpattern,1./col);
    plt.plotComplex(complexpattern, MOD, 1, 1, "autocbroad", 1);
    myIFFT(solved, complexpattern);
    applyNorm(complexpattern,1./col);
    plt.plotComplex(complexpattern, MOD, 1, 1, "autocsolved", 1);
    mwl.writeSpectra("spectra_new.txt");
    return 0;
}
