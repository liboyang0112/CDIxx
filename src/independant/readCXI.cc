#include "readCXI.hpp" 
#include "fmt/core.h"
#include <hdf5.h>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "format.hpp"
#include "cub_wrap.hpp"
#include "memManager.hpp"
using namespace std;


#define RANK         2
#define RANK_OUT     2

Real* readCXI (const char* filename, int n, Real **mask)
{
    hid_t       file, dataset, entry, imagehd, maskhd, patternhd;         /* handles */
    hid_t       datatype, dataspace;   
    hid_t       memspace; 
    H5T_class_t classm;                 /* datatype class */
    H5T_order_t order;                 /* data order */
    size_t      size;                  
    hsize_t     dims_out[2];           /* dataset dimensions */      
    int         rank;

    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    entry = H5Gopen(file, fmt::format("entry_{}", n).c_str(), H5P_DEFAULT);
    imagehd = H5Gopen(entry, "image_1", H5P_DEFAULT);
    dataset = H5Dopen(imagehd, "data",H5P_DEFAULT);

    patternhd = H5Gopen(entry, "instrument_1", H5P_DEFAULT);
    patternhd = H5Gopen(patternhd, "detector_1", H5P_DEFAULT);
    maskhd = H5Dopen(patternhd, "mask",H5P_DEFAULT);
    patternhd = H5Dopen(patternhd, "data", H5P_DEFAULT);
    fmt::println("done reading handle!");

    datatype  = H5Dget_type(dataset);     /* datatype handle */ 
    classm     = H5Tget_class(datatype);
    if (classm == H5T_INTEGER) fmt::println("Data set has INTEGER type ");
    else if(classm == H5T_COMPOUND) {
	    fmt::println("CMPOUND DATATYPE {{");
	    fmt::println(" {} bytes",static_cast<ssize_t>(H5Tget_size(datatype)));
	    fmt::println(" {} members",H5Tget_nmembers(datatype));
    }

    order     = H5Tget_order(datatype);
    if (order == H5T_ORDER_LE) fmt::println("Little endian order ");

    size  = H5Tget_size(datatype);
    fmt::println(" Data size is {} ", static_cast<ssize_t>(size));

    dataspace = H5Dget_space(dataset);    /* dataspace handle */
    rank      = H5Sget_simple_extent_ndims(dataspace);
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
    fmt::println("rank {}, dimensions {} x {} ", rank, (unsigned long)(dims_out[0]), (unsigned long)(dims_out[1]));
    myDMalloc(complexFormat, image, dims_out[0]*dims_out[1]);
    myDMalloc(Real, pattern, dims_out[0]*dims_out[1]);
    myDMalloc(int, intmask, dims_out[0]*dims_out[1]);
    hid_t complex_id = H5Tcreate(H5T_COMPOUND,sizeof(float)*2);
    H5Tinsert(complex_id,"r",0,H5T_NATIVE_FLOAT);
    H5Tinsert(complex_id,"i",sizeof(float),H5T_NATIVE_FLOAT);
    memspace = H5Screate_simple(RANK_OUT,dims_out,NULL);   
    size_t N = dims_out[0]*dims_out[1];
    init_cuda_image();
    resize_cuda_image(dims_out[0], dims_out[1]);
    plt.init(dims_out[0],dims_out[1]);

    if(mask){
      H5Dread(maskhd, H5T_STD_I32LE, memspace, dataspace,
          H5P_DEFAULT, intmask);
      myCuMalloc(Real, *mask, N);
      myCuDMalloc(int, d_mask, N);
      myMemcpyH2D(d_mask, intmask, N*sizeof(int));
      assignVal(*mask, d_mask);
      applyNorm(*mask, 1./128);
      invert(*mask);
      plt.plotFloat(*mask, MOD, 0, 1., "mask", 0, 0, 0);
    }

    H5Dread(dataset, complex_id, memspace, dataspace,
        H5P_DEFAULT, image);
    myCuDMalloc(complexFormat, d_cimage, N);
    myMemcpyH2D(d_cimage, image, N*sizeof(complexFormat));
    plt.plotComplexColor(d_cimage, 0, 1, "input");

    H5Dread(patternhd, H5T_IEEE_F32LE, memspace, dataspace,
          H5P_DEFAULT, pattern);
    myCuDMalloc(Real, d_pattern, N);
    myMemcpyH2D(d_pattern, pattern, N*sizeof(Real));
    applyNorm(d_pattern, 1./findMax(d_pattern));
    plt.saveFloat(d_pattern, "hene_pattern");
    plt.plotFloat(d_pattern, MOD, 0, 1, "input_pattern", 1, 0, 1);
    /*

       if(mask) mask = new Mat(dims_out[0],dims_out[1],CV_8UC1);
       int *maskdata = (int*)malloc(dims_out[0]*dims_out[1]*sizeof(int));
       Real noiseScale = (rcolor-1)/(rcolor-noiseLevel-1);
       for(int i = 0 ; i < image.total(); i++){
       auto &datai = ((complex<float>*)image.data)[i];
       datai-=noiseLevel;
       datai *= noiseScale;
       ((char*)imageint.data)[i] = (int)(datai.real()/256);
       ((char*)imagelog.data)[i] = datai.real() > 1?(int)(log(datai.real())*16):0;
       if(mask) ((char*)(mask)->data)[i] = maskdata[i]*255;
       }
       */
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);
    return d_pattern;
}     

