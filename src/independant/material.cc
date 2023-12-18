#include "material.hpp"
#include <math.h>
double ToyMaterial::getRefractiveIndex(double lambda){
  return 1+0.06*sin(5*M_PI*lambda);
};
double ToyMaterial::getExtinctionLength(double lambda){
  if(lambda > 1.2 && lambda < 1.21) return 0.05;
  return (1.1+sin(M_PI/2*lambda))*100;
}
double TabledMaterial::getRefractiveIndex(double lambda){
  double idx = 0;
  return idx;
}
double TabledMaterial::getExtinctionLength(double lambda){
  double len = 0;
  return len;
}
