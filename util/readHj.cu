#include"cnpy.h"
#include"cudaConfig.h"
#include"format.h"
#include"common.h"
#include<complex>
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>
#include<fstream>
#include"monoChromo.h"

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
    printf("image size = (%d, %d), spectra size = %d\n", row, col, nlambda);
    int mynlambda = int(row*(lambdasd[nlambda-1]/lambdasd[0]-1))/2;
    Real dlambda = 2./row;
    Real* myspectra = (Real*)ccmemMngr.borrowCache(sizeof(Real)*mynlambda);
    Real* mylambdas = (Real*)ccmemMngr.borrowCache(sizeof(Real)*mynlambda);
    Real nextlmd = 1;
    int ilmd = 0;
    for(int i = 0; i < nlambda; i++){
      Real currlmd = lambdasd[i]/lambdasd[0];
      if(currlmd >= nextlmd){
        mylambdas[ilmd] = nextlmd;
        nextlmd+=dlambda;
        myspectra[ilmd++] = spectrad[i];
      }
    }
    Real intensitysum = 0;
    for(int i = 0; i < mynlambda; i++){
      intensitysum+=myspectra[i];
    }
    for(int i = 0; i < mynlambda; i++){
      myspectra[i]/=intensitysum;
    }
    std::ofstream file;
    file.open("spectra.txt", std::ios::out);
    for(int i = 0; i < mynlambda; i++){
      //myspectra[i] = 0.01;
      file<<mylambdas[i]<<" "<<myspectra[i]<<std::endl;
    }
    file.close();
    Real* realb = (Real*)memMngr.borrowCache(sizeof(Real)*row*col);
    double* doubleb = (double*)memMngr.useOnsite(sizeof(double)*row*col);
    cudaMemcpy(doubleb, b.data<double>(), sizeof(double)*row*col, cudaMemcpyHostToDevice);
    init_cuda_image(row, col, 65535, 1);
    init_fft(row,col);
    cudaF(assignVal)(realb, doubleb);
    cudaF(applyNorm)(realb, 1./intensitysum);
    monoChromo mwl;
    printf("init monochrom\n");
    mwl.init(row, col, mynlambda, mylambdas, myspectra);
    plt.init(row, col);
    complexFormat* complexpattern = (complexFormat*)memMngr.borrowCache(sizeof(double)*row*col);
    complexFormat* solved = (complexFormat*)memMngr.borrowCache(sizeof(double)*row*col);
    cudaF(extendToComplex)(realb, complexpattern);
    plt.plotComplex(complexpattern,REAL,0,1,"logbroadpattern",1);
    plt.plotComplex(complexpattern,REAL,0,1,"broadpattern",0);
    printf("solving matrix\n");
    mwl.solveMWL(complexpattern, solved, 0, 30, 1, 0);
    for(int i = 0; i < mynlambda; i++){
      myspectra[i] = 0.01;
    }
    mwl.solveMWL(complexpattern, solved, solved, 300, 0, 1);
    plt.plotComplex(solved,REAL,0,1,"logmonopattern",1);
    plt.plotComplex(solved,REAL,0,1,"monopattern",0);
    cudaF(getMod)(realb, solved);
    plt.saveFloat(realb, "pattern");
    myCufftExec( *plan, complexpattern, complexpattern, CUFFT_INVERSE);
    cudaF(applyNorm)(complexpattern,1./col);
    plt.plotComplex(complexpattern, MOD, 1, 1, "autocbroad", 1);
    myCufftExec( *plan, solved, complexpattern, CUFFT_INVERSE);
    cudaF(applyNorm)(complexpattern,1./col);
    plt.plotComplex(complexpattern, MOD, 1, 1, "autocsolved", 1);
    file.open("spectra_new.txt", std::ios::out);
    for(int i = 0; i < mynlambda; i++){
      file<<mylambdas[i]<<" "<<myspectra[i]<<std::endl;
    }
    file.close();

    return 0;
}
