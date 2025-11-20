#include <cstring>
#include <string>
#include <math.h>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "fdtd3d.hpp"
using namespace std;


int main(){
  int nsteps = 1000;
  const int nx = 150;
  const int ny = 150;
  const int nz = 150;
  //dim3 nblkx,nblky,nblkz, nthd2d;
  ////---------inner dimensions--------
  ////-----boundary dimensions---------
  //nthd2d.x = 256;
  //nblkx.x = (ny*nz-1)/nthd2d.x+1;
  //nblky.x = (nx*nz-1)/nthd2d.x+1;
  //nblkz.x = (nx*ny-1)/nthd2d.x+1;

  size_t nnode = nx*ny*nz;
  size_t memsz = nnode*sizeof(Real);
  Real* Hz = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Hx = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Hy = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ex = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ey = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ez = (Real*)memMngr.borrowCleanCache(memsz);
  //Real* Px = (Real*)memMngr.borrowCleanCache(memsz);  //polarization
  //Real* Py = (Real*)memMngr.borrowCleanCache(memsz);
  //Real* Pz = (Real*)memMngr.borrowCleanCache(memsz);
  //Real* dPx = (Real*)memMngr.borrowCleanCache(memsz); //time derivative of polarization
  //Real* dPy = (Real*)memMngr.borrowCleanCache(memsz);
  //Real* dPz = (Real*)memMngr.borrowCleanCache(memsz);

  //unsigned char* material_map = (unsigned char*)memMngr.borrowCleanCache(nnode);  //supports 255 kinds of materials
  //record boundaries for PML
  myCuDMallocClean(Real, EzBdx1, ny*nz);
  myCuDMallocClean(Real, EyBdx1, ny*nz);
  myCuDMallocClean(Real, EzBdy1, nz*nx);
  myCuDMallocClean(Real, ExBdy1, nz*nx);
  myCuDMallocClean(Real, ExBdz1, nx*ny);
  myCuDMallocClean(Real, EyBdz1, nx*ny);
  myCuDMallocClean(Real, HzBdx0, ny*nz);
  myCuDMallocClean(Real, HyBdx0, ny*nz);
  myCuDMallocClean(Real, HzBdy0, nz*nx);
  myCuDMallocClean(Real, HxBdy0, nz*nx);
  myCuDMallocClean(Real, HxBdz0, nx*ny);
  myCuDMallocClean(Real, HyBdz0, nx*ny);
  //select slice for visualization
  Real* slice = (Real*)memMngr.borrowCache(nx*ny*sizeof(Real));

  bool saveField = 0;
  int sourcePos = nx/2+nx*ny/2 + nx*ny*nz/2;
  resize_cuda_image(nx,ny);
  plt.init(nx,ny);
  init_cuda_image();
  int ezvid = plt.initVideo("Ez.mp4", 24);
  int hxvid = plt.initVideo("Hx.mp4", 24);
  int hyvid = plt.initVideo("Hy.mp4", 24);
  plt.showVid = -1;//ezvid;
  for(int i = 0; i < nsteps; i++){
    saveField = i%5==0;
    //applySource<<<1,1>>>(Ez, sourcePos, 20*sin(M_PI/70*i));//50*exp(-sq(double(i-100)/30)));
    if(i < 280) {
      applySource(Ez, sourcePos, 30*exp(-sq((i-140.)/70))*(sin(M_PI/35*i)));//50*exp(-sq(double(i-100)/30,2)));
      applySource(Hx, sourcePos, 30*exp(-sq((i-140.)/70))*(sin(M_PI/35*i))*(0.4/0.5)/3);//50*exp(-sq(double(i-100)/30,2)));
      applySource(Hx, sourcePos+1, 30*exp(-sq((i-140.)/70))*(sin(M_PI/35*i))*(0.4/0.5)/3);//50*exp(-sq(double(i-100)/30,2)));
      applySource(Hx, sourcePos-1, 30*exp(-sq((i-140.)/70))*(sin(M_PI/35*i))*(0.4/0.5)/3);//50*exp(-sq(double(i-100)/30,2)));
      //applySource<<<1,1>>>(Ey, sourcePos, 500*exp(-sq((i-140.)/70))*(cos(M_PI/35*i)));//50*exp(-sq(double(i-100)/30,2)));
    }//get circular polarized source!
    //applySourceV<<<nblkx,nthd2d>>>(Ez, nx, ny, nz, 100, 5*sin(M_PI/30*i));

    resize_cuda_image(ny,nz);
    applyPMLx1(Hx, Hy, Hz, Ex, Ey, Ez, EyBdx1, EzBdx1, nx);
    //applyPMLx1post(Hx, Hy, Hz, Ex, Ey, Ez, ExBdx1, EyBdx1, EzBdx1, nx);
    resize_cuda_image(n_PML,ny,nz);
    applyPMLx1_d(Hx, Hy, Hz, Ex, Ey, Ez, nx);  //do separately to speed up
    resize_cuda_image(nx,nz);
    applyPMLy1(Hx, Hy, Hz, Ex, Ey, Ez, ExBdy1, EzBdy1, ny);
    resize_cuda_image(nx,ny);
    applyPMLz1(Hx, Hy, Hz, Ex, Ey, Ez, ExBdz1, EyBdz1, nz);

    //applyPeriodicx_E<<<nblkx, nthd2d>>>(Ex, Ey, Ez, nx, ny, nz);
    //applyPeriodicy_E<<<nblky, nthd2d>>>(Ex, Ey, Ez, nx, ny, nz);
    //applyPeriodicz_E<<<nblky, nthd2d>>>(Ex, Ey, Ez, nx, ny, nz);

    resize_cuda_image(nx,ny,nz);
    updateH(Hx, Hy, Hz, Ex, Ey, Ez);    //------------UPDATE H-----------
    resize_cuda_image(nx,ny);

    //applyPeriodicx_H<<<nblkx, nthd2d>>>(Hx, Hy, Hz, nx, ny, nz);
    //applyPeriodicy_H<<<nblkx, nthd2d>>>(Hx, Hy, Hz, nx, ny, nz);
    //applyPeriodicz_H<<<nblkx, nthd2d>>>(Hx, Hy, Hz, nx, ny, nz);

    resize_cuda_image(ny, nz);
    applyPMLx0(Hx, Hy, Hz, Ex, Ey, Ez, HyBdx0, HzBdx0, nx);
    resize_cuda_image(nx, ny, n_PML);
    applyPMLx0_d(Hx, Hy, Hz, Ex, Ey, Ez);
    applyPMLy0(Hx, Hy, Hz, Ex, Ey, Ez, HxBdy0, HzBdy0, ny);
    resize_cuda_image(nx,ny);
    applyPMLz0(Hx, Hy, Hz, Ex, Ey, Ez, HxBdz0, HyBdz0);

    resize_cuda_image(nx,ny,nz);
    updateE(Hx, Hy, Hz, Ex, Ey, Ez);  //------------UPDATE E-----------
    resize_cuda_image(nx,ny);
    if(saveField) {
      getXYSlice(slice, Ez, nx, ny, nz/2);
      //getXZSlice(slice, Ey , nx, ny, nz, 100);
      plt.toVideo = ezvid;
      plt.plotFloat(slice, REAL, 0, 10, "",0,0,1, ("Ez,t="+to_string(i)).c_str());
      getXYSlice(slice, Hx , nx, ny, nz/2);
      plt.toVideo = hxvid;
      plt.plotFloat(slice, REAL, 0, 10, "",0,0,1,("Hx,t="+to_string(i)).c_str());
      getXYSlice(slice, Hy , nx, ny, nz/2);
      plt.toVideo = hyvid;
      plt.plotFloat(slice, REAL, 0, 100, "",0,0,1,("Hy,t="+to_string(i)).c_str());
    }
  }
  plt.saveVideo(ezvid);
  plt.saveVideo(hxvid);
  plt.saveVideo(hyvid);
}
