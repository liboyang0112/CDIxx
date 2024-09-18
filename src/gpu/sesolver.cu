#include "cudaDefs_h.cu"
cuFunc(initV,(Real* V, Real val), (V, val), {
  cuda3Idx()
  int &nx = cuda_row;
  int &ny = cuda_column;
  int &nz = cuda_height;
  //Real xmid = nx/2-0.5-2;
  //Real ymid = ny/2-0.5-1;
  //Real zmid = nz/2-0.5;
  //Real r2 = sq(x-xmid) + sq(y-ymid) + sq(z-zmid);
  //V[index] = -val/sqrt(r2);
  Real xmid1 = nx/3-0.5;
  Real ymid1 = ny/3-0.5;
  Real zmid1 = nz/2-0.5;
  Real xmid2 = nx*2/3-0.5;
  Real ymid2 = ny*2/3-0.5;
  Real zmid2 = nz/2-0.5;
  Real r21 = sq(x-xmid1) + sq(y-ymid1) + sq(z-zmid1);
  Real r22 = sq(x-xmid2) + sq(y-ymid2) + sq(z-zmid2);
  V[index] += val/sqrt(r21) + val/sqrt(r22);
  if(r21 < 20 || r22 < 20) printf("V=%f\n", V[index]);
})

cuFunc(Hpsifunc, (Real * psi, Real *V, Real *Hpsi, Real Eshift),
       (psi, V, Hpsi, Eshift), {
         cuda3Idx();
         Hpsi[index] = (Eshift + 6 + V[index]) * psi[index];
         if (z < cuda_height - 1) {
           Hpsi[index] -= psi[index + cuda_row * cuda_column];
         }
         if (z > 0) {
           Hpsi[index] -= psi[index - cuda_row * cuda_column];
         }
         if (x < cuda_row - 1) {
           Hpsi[index] -= psi[index + 1];
         }
         if (x > 0) {
           Hpsi[index] -= psi[index - 1];
         }
         if (y < cuda_column - 1) {
           Hpsi[index] -= psi[index + cuda_row];
         }
         if (y > 0) {
           Hpsi[index] -= psi[index - cuda_row];
         }
       });
cuFunc(getXZSlice, (Real * slice, Real *data, int nx, int ny, int nz, int iy),
       (slice, data, nx, ny, nz, iy), {
         int index = blockIdx.x * blockDim.x + threadIdx.x;
         if (index >= nx * nz)
           return;
         int x = index % nx;
         int z = index / nx;
         slice[index] = data[x + nx * iy + nx * ny * z];
       });
cuFunc(getYZSlice, (Real * slice, Real *data, int nx, int ny, int nz, int ix),
       (slice, data, nx, ny, nz, ix), {
         int index = blockIdx.x * blockDim.x + threadIdx.x;
         if (index >= ny * nz)
           return;
         int y = index % ny;
         int z = index / ny;
         int idx = ix + nx * y + nx * ny * z;
         slice[index] = data[idx];
       })
