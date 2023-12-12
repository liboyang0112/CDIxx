#include "material.h"
#include <math.h>
double ToyMaterial::getRefractiveIndex(double lambda){
  return 1+0.006*sin(5*M_PI*lambda);
};
double ToyMaterial::getExtinctionLength(double lambda){
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
