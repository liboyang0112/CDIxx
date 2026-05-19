#include<vector>
#include<string>
#include"format.hpp"
class material {
public:
  Real* deltas;
  Real* betas;
  Real* deltas_d;
  Real* betas_d;
  double* lambdas_d;
  Real lambda_ref = 1;
  int ne;
  int nl;
  material(){};
  void init(const std::vector<std::string>& fnames, double* lambdas, int nlambda, Real lambda_ref_ = 1);
  void Transmission(complexFormat* data, Real* maps, int npix, Real thickness);
  ~material();
};
void* computeTransmission(complexFormat* out,Real* maps,Real* deltas,Real* betas,double* lambdas,int ne,int nl, int npix, Real lambda_ref, Real thickness);
