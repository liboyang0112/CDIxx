#include<vector>
#include<string>
#include"format.hpp"
class material {
public:
  Real* deltas;
  Real* betas;
  Real* deltas_d;
  Real* betas_d;
  int ne;
  int nl;
  double to_eV(int l) const { return 1239.84193 / static_cast<double>(l); }
  material(const std::vector<std::string>& fnames, const std::vector<int>& lambdas);
  complexFormat* computeTransmission(Real* maps);
  ~material();
};
void* computeTransmission(complexFormat* out,Real* maps,Real* deltas,Real* betas,Real* lambdas,int ne,int nl);
