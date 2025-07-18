#include "fmt/core.h"
#include "format.hpp"
#include <string.h>
#include <stdio.h>

//void tqli(Real *diag, Real *subdiag, int n, Real *eigenVectors) {
//    for(int l = 1; l < n; l++){
//        
//    }
//}
int main() {
  const int n = 100;
  Real diagEle[n], subdiagEle[n], eigenVectors[n * n];
  memset(eigenVectors, 0, n * n * sizeof(Real));
  for (int i = 0; i < n; i++) {
    diagEle[i] = -2;
    subdiagEle[i] = 1;
    eigenVectors[i * n + i] = 1;
  }
  subdiagEle[n-1] = 0;
//  tqli(diagEle, subdiagEle, n, eigenVectors);
  for (int i = 0; i < n; i++) {
    fmt::print("{:f}", eigenVectors[i * n + n - 1]);
  }
  return 0;
}
