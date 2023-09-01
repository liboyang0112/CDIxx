#include <eigen3/Eigen/Dense>
using namespace Eigen;
using namespace std;
int inverseMatrixEigen(double* matrix, int n)
{
  Map<MatrixXd> m(matrix, n, n);
  m = m.inverse();
  return 0;
}
