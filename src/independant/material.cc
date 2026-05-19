#include <fmt/base.h>
#include <material.hpp>
#include "cudaConfig.hpp"
#include<fstream>
#include<gsl/gsl_spline.h>
#include<gsl/gsl_interp.h>

double to_eV(double l) { return 1239.84193 / l; }
void material::init(const std::vector<std::string>& fnames, double* lambdas, int nlambda, Real lambda_ref_) {
    lambda_ref = lambda_ref_;
    ne = fnames.size();
    nl = nlambda;
    deltas = new Real[ne * nl];
    betas = new Real[ne * nl];
    gsl_interp_accel* acc_d = gsl_interp_accel_alloc();
    gsl_interp_accel* acc_b = gsl_interp_accel_alloc();
    gsl_spline* sp_d = nullptr;
    gsl_spline* sp_b = nullptr;
    for(int i = 0; i < ne; ++i) {
      std::ifstream fin(fnames[i]);
      std::string line1, line2;
      std::getline(fin, line1);
      std::getline(fin, line2);
      size_t pos = line1.find("Density=");
      double density = 1.0;
      if(pos != std::string::npos) density = std::stod(line1.substr(pos + 8));
      std::vector<double> en, dl, bt;
      double e, d, b;
      while(fin >> e >> d >> b) {
        en.push_back(e);
        dl.push_back(d / density);
        bt.push_back(b / density);
      }
      fin.close();
      int n = en.size();
      if(n > 1) {
        sp_d = gsl_spline_alloc(gsl_interp_cspline, n);
        sp_b = gsl_spline_alloc(gsl_interp_cspline, n);
        gsl_spline_init(sp_d, en.data(), dl.data(), n);
        gsl_spline_init(sp_b, en.data(), bt.data(), n);
        double emin = en.front();
        double emax = en.back();
        for(int j = 0; j < nl; ++j) {
          double qe = to_eV(lambdas[j]*lambda_ref*1e3);
          if(qe < emin) qe = emin;
          if(qe > emax) qe = emax;
          deltas[i * nl + j] = gsl_spline_eval(sp_d, qe, acc_d);
          betas[i * nl + j] = gsl_spline_eval(sp_b, qe, acc_b);
        }
        gsl_spline_free(sp_d);
        gsl_spline_free(sp_b);      }
    }
    gsl_interp_accel_free(acc_d);
    gsl_interp_accel_free(acc_b);
    myCuMalloc(Real, deltas_d, nl*ne);
    myCuMalloc(Real, betas_d, nl*ne);
    myCuMalloc(double, lambdas_d, nlambda);
    myMemcpyH2D(deltas_d, deltas, nl*ne*sizeof(Real));
    myMemcpyH2D(betas_d, betas, nl*ne*sizeof(Real));
    myMemcpyH2D(lambdas_d, lambdas, nlambda*sizeof(double));
  }
material::~material() {
    delete[] deltas;
    myCuFree(deltas_d);
    myCuFree(betas_d);
    delete[] betas;
  }
void material::Transmission(complexFormat* data, Real* maps, int npix, Real thickness){
  computeTransmission(data,maps,deltas_d,betas_d,lambdas_d,ne,nl,npix,lambda_ref, thickness);
}
