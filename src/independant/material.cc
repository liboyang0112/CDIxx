#include <material.hpp>
#include "cudaConfig.hpp"
#include<fstream>
#include<gsl/gsl_spline.h>
#include<gsl/gsl_interp.h>
material::material(const std::vector<std::string>& fnames, const std::vector<int>& lambdas) : ne(fnames.size()), nl(lambdas.size()) {
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
          double qe = to_eV(lambdas[j]);
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
    myMemcpyH2D(deltas_d, deltas, nl*ne);
    myMemcpyH2D(betas_d, betas, nl*ne);
  }
material::~material() {
    delete[] deltas;
    myCuFree(deltas_d);
    myCuFree(betas_d);
    delete[] betas;
  }
