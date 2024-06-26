#include"cudaConfig.hpp"
#include"FGA.hpp"
#include"cuPlotter.hpp"
#include"monoChromo.hpp"
#include"cub_wrap.hpp"

int FGA(int row, int col, int nlambda, double* lambdas, double* spectra, Real* data, int niter, int binskip)
{
    Real* realb = (Real*)memMngr.borrowCache(sizeof(Real)*row*col);
    myMemcpyH2D(realb, data, sizeof(Real)*row*col);
    init_cuda_image();
    resize_cuda_image(row, col);
    applyNorm(realb, 1./findMax(realb));
    monoChromo mwl;
    //monoChromo_constRatio mwl;
    mwl.jump = binskip;
    mwl.skip = mwl.jump/2;
    mwl.init(row, col, lambdas, spectra, nlambda);
    plt.init(row, col);
    complexFormat* complexpattern = (complexFormat*)memMngr.borrowCache(sizeof(double)*row*col);
    complexFormat* solved = (complexFormat*)memMngr.borrowCache(sizeof(double)*row*col);
    extendToComplex(realb, complexpattern);
    plt.saveFloat(realb, "broadpattern");
    plt.plotComplex(complexpattern,MOD,0,1,"logbroadpattern",1,0,1);
    plt.plotComplex(complexpattern,MOD,0,1,"broadpattern",0);
    mwl.solveMWL(complexpattern, solved, 0, 1, niter, 1, 0);
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
    mwl.writeSpectra("spectra_new.txt", lambdas[0]);
    return 0;
}
