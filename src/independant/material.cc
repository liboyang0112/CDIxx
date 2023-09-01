#include "material.h"
#include <math.h>
double ToyMaterial::getRefractiveIndex(double lambda){
  return 1+0.2*sin(lambda/100);
};
double ToyMaterial::getExtinctionLength(double lambda){
  return (1.1+sin(lambda/100))*1e-5;
}
double TabledMaterial::getRefractiveIndex(double lambda){
  double idx = 0;
  return idx;
}
double TabledMaterial::getExtinctionLength(double lambda){
  double len = 0;
  return len;
}
